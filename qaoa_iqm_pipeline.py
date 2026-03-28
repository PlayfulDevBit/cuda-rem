"""
qaoa_iqm_pipeline.py — Prefect Tasks and Flow
===============================================
YOU DO NOT NEED TO EDIT THIS FILE.

If you are a researcher working on a custom REM algorithm, edit rem.py instead.
This file imports calibrate() and apply() from rem.py automatically.

Pipeline steps:
    1. Configure IQM Resonance target
    2. Build problem graph (MaxCut)
    3. Build cost Hamiltonian
    4. Calibrate readout noise          ← uses rem.calibrate()
    5. Run COBYLA optimisation loop     ← uses rem.apply() each iteration
    6. Post-process and report results
"""

import os
import json
import itertools
from typing import Optional

import numpy as np
import networkx as nx
from scipy.optimize import minimize
import cudaq

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

# REM is imported from rem.py — swap rem.py to change the algorithm
from rem import calibrate as rem_calibrate
from rem import apply as rem_apply


# ─────────────────────────────────────────────
# 1.  IQM RESONANCE TARGET
# ─────────────────────────────────────────────

@task(name="configure-iqm-target", retries=2, retry_delay_seconds=5)
def configure_iqm_target():
    log = get_run_logger()
    token = os.environ.get("IQM_TOKEN")
    server_url = os.environ.get("IQM_SERVER_URL")

    if os.environ.get("CUDAQ_TARGET") == "qpp-cpu":
        cudaq.set_target("qpp-cpu")
        log.info("Target set → qpp-cpu (local simulator)")
        return "qpp-cpu"

    if not token:
        raise EnvironmentError("IQM_TOKEN environment variable is not set.")
    if not server_url:
        raise EnvironmentError("IQM_SERVER_URL environment variable is not set.")

    cudaq.set_target("iqm", url=server_url, **{"credentials": token})
    log.info(f"Target set → IQM Resonance  |  {server_url}")
    return "iqm"


# ─────────────────────────────────────────────
# 2.  PROBLEM GRAPH
# ─────────────────────────────────────────────

@task(name="build-problem-graph")
def build_problem_graph(n_qubits: int, seed: int) -> dict:
    log = get_run_logger()
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n_qubits, p=0.6, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.uniform(0.5, 1.5))

    edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
    log.info(f"Graph: {n_qubits} nodes, {len(edges)} edges")

    create_markdown_artifact(
        key="problem-graph",
        markdown=f"## Problem Graph\n- **Nodes:** {n_qubits}\n- **Edges:** {edges}",
        description="MaxCut problem graph",
    )
    return {"n_qubits": n_qubits, "edges": edges}


# ─────────────────────────────────────────────
# 3.  COST HAMILTONIAN
# ─────────────────────────────────────────────

@task(name="build-cost-hamiltonian")
def build_cost_hamiltonian(graph: dict) -> cudaq.SpinOperator:
    log = get_run_logger()
    n = graph["n_qubits"]
    H = cudaq.SpinOperator.from_word("I" * n) * 0.0
    for u, v, w in graph["edges"]:
        H += w * 0.5 * (
            cudaq.SpinOperator.from_word("I" * n) - _zz_term(n, u, v)
        )
    log.info(f"Hamiltonian: {H.get_term_count()} terms")
    return H


def _zz_term(n, i, j):
    word = ["I"] * n
    word[i] = "Z"
    word[j] = "Z"
    return cudaq.SpinOperator.from_word("".join(word))


# ─────────────────────────────────────────────
# 4.  READOUT CALIBRATION  (calls rem.calibrate)
# ─────────────────────────────────────────────

@task(name="calibrate-readout", retries=3, retry_delay_seconds=10)
def calibrate_readout(n_qubits: int, shots: int):
    log = get_run_logger()
    log.info("Running readout calibration via rem.calibrate()…")
    cal = rem_calibrate(n_qubits=n_qubits, shots=shots)
    log.info("Calibration complete.")
    return cal


# ─────────────────────────────────────────────
# 5.  QAOA CIRCUIT
# ─────────────────────────────────────────────

def make_qaoa_kernel(n_qubits: int, p: int, edges: list):
    @cudaq.kernel
    def qaoa(params: list[float]):
        q = cudaq.qvector(n_qubits)
        h(q)
        for layer in range(p):
            gamma = params[layer]
            beta  = params[p + layer]
            for u, v, w in edges:
                x.ctrl(q[u], q[v])
                rz(2.0 * gamma * w, q[v])
                x.ctrl(q[u], q[v])
            for i in range(n_qubits):
                rx(2.0 * beta, q[i])
    return qaoa


# ─────────────────────────────────────────────
# 6.  COBYLA OPTIMISATION  (calls rem.apply each iter)
# ─────────────────────────────────────────────

@task(name="run-cobyla-optimization", retries=1, retry_delay_seconds=30)
def run_cobyla_optimization(
    graph: dict, hamiltonian, cal, p: int, shots: int, max_iter: int, rhobeg: float
) -> dict:
    log = get_run_logger()
    n = graph["n_qubits"]
    edges = graph["edges"]
    kernel = make_qaoa_kernel(n, p, edges)
    energy_history = []

    def objective(params):
        raw = cudaq.sample(kernel, params.tolist(), shots_count=shots)
        raw_dict = {bs: cnt for bs, cnt in raw.items()}

        # ── REM correction (from rem.py) ──
        corrected = rem_apply(
            raw_counts=raw_dict, calibration=cal, n_qubits=n, shots=shots
        )

        cost = _estimate_maxcut_cost(corrected, edges)
        energy_history.append(float(cost))
        log.debug(f"  iter {len(energy_history):3d}  cost={cost:.5f}")
        return cost

    rng = np.random.default_rng(42)
    x0 = rng.uniform(-0.1, 0.1, size=2 * p)

    log.info(f"Starting COBYLA — p={p}, max_iter={max_iter}, shots={shots}")
    result = minimize(
        objective, x0, method="COBYLA",
        options={"maxiter": max_iter, "rhobeg": rhobeg, "disp": True},
    )
    log.info(f"COBYLA done — converged={result.success}  cost={result.fun:.5f}")

    return {
        "optimal_params": result.x.tolist(),
        "optimal_cost": float(result.fun),
        "n_iter": len(energy_history),
        "energy_history": energy_history,
        "converged": bool(result.success),
    }


def _estimate_maxcut_cost(probs, edges):
    cost = 0.0
    for bs, prob in probs.items():
        bits = [int(b) for b in bs]
        cut = sum(w for u, v, w in edges if bits[u] != bits[v])
        cost -= prob * cut
    return cost


# ─────────────────────────────────────────────
# 7.  POST-PROCESSING
# ─────────────────────────────────────────────

@task(name="post-process-results")
def post_process_results(graph, opt_result, cal, shots, p) -> dict:
    log = get_run_logger()
    n = graph["n_qubits"]
    edges = graph["edges"]
    kernel = make_qaoa_kernel(n, p, edges)

    log.info("Final high-shot execution at optimal parameters…")
    raw = cudaq.sample(kernel, opt_result["optimal_params"], shots_count=shots * 2)
    raw_dict = {bs: cnt for bs, cnt in raw.items()}
    corrected = rem_apply(
        raw_counts=raw_dict, calibration=cal, n_qubits=n, shots=shots * 2
    )

    best_bs = max(corrected, key=corrected.get)
    best_cut = sum(w for u, v, w in edges if best_bs[u] != best_bs[v])
    bf_cut, bf_bs = _brute_force_maxcut(n, edges)
    ar = best_cut / bf_cut if bf_cut > 0 else 0.0

    log.info(f"Best: {best_bs}  cut={best_cut:.3f}  |  BF: {bf_bs}  cut={bf_cut:.3f}")
    log.info(f"Approximation ratio: {ar:.4f}")

    top5 = sorted(corrected.items(), key=lambda x: -x[1])[:5]
    rows = "\n".join(f"| `{bs}` | {p:.4f} |" for bs, p in top5)
    create_markdown_artifact(
        key="qaoa-results",
        markdown=f"""## QAOA Results
| Metric | Value |
|--------|-------|
| Best bitstring | `{best_bs}` |
| QAOA cut value | {best_cut:.4f} |
| Optimal cut (BF) | {bf_cut:.4f} |
| Approximation ratio | {ar:.4f} |
| COBYLA converged | {opt_result['converged']} |
| Iterations | {opt_result['n_iter']} |

### Top-5 Bitstrings (REM-corrected)
| Bitstring | Probability |
|-----------|-------------|
{rows}
""",
        description="QAOA final results",
    )

    return {
        "best_bitstring": best_bs,
        "best_cut_value": best_cut,
        "optimal_cut_value": bf_cut,
        "approximation_ratio": ar,
        "corrected_distribution": corrected,
        "optimization": opt_result,
    }


def _brute_force_maxcut(n, edges):
    best, best_bs = 0.0, "0" * n
    for bits in itertools.product([0, 1], repeat=n):
        cut = sum(w for u, v, w in edges if bits[u] != bits[v])
        if cut > best:
            best, best_bs = cut, "".join(map(str, bits))
    return best, best_bs


# ─────────────────────────────────────────────
# 8.  PREFECT FLOW
# ─────────────────────────────────────────────

@flow(
    name="qaoa-iqm-resonance",
    description="QAOA on IQM Resonance — COBYLA + REM (from rem.py)",
)
def qaoa_iqm_flow(config: Optional[dict] = None):
    from run import DEFAULT_CONFIG
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    log = get_run_logger()
    log.info(f"Config: {json.dumps(cfg, indent=2)}")

    configure_iqm_target()
    graph       = build_problem_graph(cfg["n_qubits"], cfg["seed"])
    hamiltonian = build_cost_hamiltonian(graph)
    cal         = calibrate_readout(cfg["n_qubits"], cfg["rem_shots"])
    opt_result  = run_cobyla_optimization(
        graph=graph, hamiltonian=hamiltonian, cal=cal,
        p=cfg["p_layers"], shots=cfg["shots"],
        max_iter=cfg["cobyla_max_iter"], rhobeg=cfg["cobyla_rhobeg"],
    )
    final = post_process_results(
        graph=graph, opt_result=opt_result, cal=cal,
        shots=cfg["shots"], p=cfg["p_layers"],
    )

    log.info("✅  Pipeline complete.")
    return final
