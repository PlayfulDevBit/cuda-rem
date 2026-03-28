"""
qaoa_iqm_pipeline.py — QAOA on IQM Resonance
==============================================
Stack:  CUDA-Q  |  IQM Resonance (cloud)  |  Prefect 3.x
Algo:   QAOA (MaxCut)
Opt:    COBYLA (gradient-free)
Mitig:  Readout Error Mitigation — implemented in rem.py

YOU DO NOT NEED TO EDIT THIS FILE.
If you are a researcher working on a custom REM algorithm, edit rem.py instead.
This file imports calibrate() and apply() from rem.py automatically.

Local:
    python qaoa_iqm_pipeline.py
    python qaoa_iqm_pipeline.py --n_qubits 4 --p_layers 2 --shots 4096

Serverless:
    Triggered from Prefect Cloud UI after running deploy_qaoa.py

Prerequisites:
    Prefect Secret block named 'iqm-resonance-token' containing your IQM API key:
        from prefect.blocks.system import Secret
        Secret(value="your-iqm-key").save("iqm-resonance-token")
"""

import argparse
import itertools
import json
import math
import os

import networkx as nx
import numpy as np
from scipy.optimize import minimize

import cudaq

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

# REM is imported from rem.py — swap rem.py to change the algorithm
from rem import calibrate as rem_calibrate
from rem import apply as rem_apply


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def get_iqm_token() -> str:
    try:
        from prefect.blocks.system import Secret
        return Secret.load("iqm-resonance-token").get()
    except Exception:
        return os.environ.get("IQM_TOKEN", "")


def get_iqm_server_url() -> str:
    return os.environ.get("IQM_SERVER_URL", "https://cocos.resonance.meetiqm.com/garnet")


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1 — CONFIGURE IQM TARGET
# ═══════════════════════════════════════════════════════════════════════

@task(name="1 · Configure IQM Target", retries=2, retry_delay_seconds=5)
def configure_iqm_target() -> str:
    log = get_run_logger()

    token = get_iqm_token()
    server_url = get_iqm_server_url()

    if not token:
        log.warning("IQM token not found — falling back to local CPU simulator (qpp-cpu).")
        cudaq.set_target("qpp-cpu")
        return "qpp-cpu"

    cudaq.set_target("iqm", url=server_url, **{"credentials": token})
    log.info(f"CUDA-Q target set → IQM Resonance  |  {server_url}")
    return "iqm"


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2 — BUILD PROBLEM GRAPH
# ═══════════════════════════════════════════════════════════════════════

@task(name="2 · Build Problem Graph")
def build_problem_graph(n_qubits: int, seed: int) -> dict:
    """
    Build a random weighted MaxCut graph.
    Swap this task for your own problem Hamiltonian if needed.
    """
    log = get_run_logger()
    rng = np.random.default_rng(seed)

    G = nx.erdos_renyi_graph(n_qubits, p=0.6, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.uniform(0.5, 1.5))

    edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
    log.info(f"Graph: {n_qubits} nodes, {len(edges)} edges")

    create_markdown_artifact(
        key="qaoa-problem-graph",
        markdown=(
            f"## Problem Graph\n"
            f"- **Nodes (qubits):** {n_qubits}\n"
            f"- **Edges:** {len(edges)}\n"
            f"- **Edge list:** {edges}\n"
        ),
        description="MaxCut problem graph",
    )

    return {"n_qubits": n_qubits, "edges": edges}


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3 — BUILD COST HAMILTONIAN
# ═══════════════════════════════════════════════════════════════════════

@task(name="3 · Build Cost Hamiltonian")
def build_cost_hamiltonian(graph: dict):
    log = get_run_logger()
    n = graph["n_qubits"]

    H = cudaq.SpinOperator.from_word("I" * n) * 0.0
    for u, v, w in graph["edges"]:
        H += w * 0.5 * (
            cudaq.SpinOperator.from_word("I" * n) - _zz_term(n, u, v)
        )

    log.info(f"Hamiltonian built — {H.get_term_count()} terms")
    return H


def _zz_term(n: int, i: int, j: int):
    word = ["I"] * n
    word[i] = "Z"
    word[j] = "Z"
    return cudaq.SpinOperator.from_word("".join(word))


# ═══════════════════════════════════════════════════════════════════════
# STAGE 4 — READOUT CALIBRATION  (calls rem.calibrate)
# ═══════════════════════════════════════════════════════════════════════

@task(name="4 · Calibrate Readout", retries=3, retry_delay_seconds=10)
def calibrate_readout(n_qubits: int, shots: int):
    log = get_run_logger()
    log.info("Running readout calibration via rem.calibrate()…")
    cal = rem_calibrate(n_qubits=n_qubits, shots=shots)
    log.info("Calibration complete.")
    return cal


# ═══════════════════════════════════════════════════════════════════════
# STAGE 5 — QAOA CIRCUIT KERNEL
# ═══════════════════════════════════════════════════════════════════════

def make_qaoa_kernel(n_qubits: int, p: int, edges: list):
    """
    Factory: returns a CUDA-Q kernel for QAOA with p layers.
    Parameters layout: [gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}]
    """
    @cudaq.kernel
    def qaoa(params: list[float]):
        q = cudaq.qvector(n_qubits)

        # |+>^n initial state
        h(q)

        for layer in range(p):
            gamma = params[layer]
            beta  = params[p + layer]

            # Cost unitary: exp(-i gamma H_C)
            for u, v, w in edges:
                x.ctrl(q[u], q[v])
                rz(2.0 * gamma * w, q[v])
                x.ctrl(q[u], q[v])

            # Mixer unitary: exp(-i beta B)
            for i in range(n_qubits):
                rx(2.0 * beta, q[i])

    return qaoa


# ═══════════════════════════════════════════════════════════════════════
# STAGE 6 — COBYLA OPTIMISATION  (calls rem.apply each iteration)
# ═══════════════════════════════════════════════════════════════════════

@task(name="6 · COBYLA Optimisation", retries=1, retry_delay_seconds=30)
def run_cobyla_optimization(
    graph: dict,
    cal,
    p: int,
    shots: int,
    max_iter: int,
    rhobeg: float,
) -> dict:
    """
    COBYLA outer loop. Each objective evaluation:
      1. Runs QAOA circuit on IQM Resonance via CUDA-Q
      2. Applies REM to raw counts  (via rem.apply)
      3. Estimates <H_C> from corrected probabilities
    """
    log = get_run_logger()
    n = graph["n_qubits"]
    edges = graph["edges"]
    kernel = make_qaoa_kernel(n, p, edges)
    energy_history = []

    def objective(params: np.ndarray) -> float:
        raw = cudaq.sample(kernel, params.tolist(), shots_count=shots)
        raw_dict = {bs: cnt for bs, cnt in raw.items()}

        # REM correction (from rem.py)
        corrected = rem_apply(
            raw_counts=raw_dict, calibration=cal, n_qubits=n, shots=shots
        )

        cost = _estimate_maxcut_cost(corrected, edges)
        energy_history.append(float(cost))
        log.debug(f"  iter {len(energy_history):3d}  cost={cost:.5f}")
        return cost  # COBYLA minimises — MaxCut negated inside _estimate_maxcut_cost

    rng = np.random.default_rng(42)
    x0 = rng.uniform(-0.1, 0.1, size=2 * p)

    log.info(f"Starting COBYLA — p={p}, max_iter={max_iter}, shots={shots}")
    result = minimize(
        objective, x0,
        method="COBYLA",
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


def _estimate_maxcut_cost(probs: dict, edges: list) -> float:
    cost = 0.0
    for bs, prob in probs.items():
        bits = [int(b) for b in bs]
        cut = sum(w for u, v, w in edges if bits[u] != bits[v])
        cost -= prob * cut  # negative: COBYLA minimises
    return cost


# ═══════════════════════════════════════════════════════════════════════
# STAGE 7 — POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════════

@task(name="7 · Post-Process Results")
def post_process_results(
    graph: dict,
    opt_result: dict,
    cal,
    shots: int,
    p: int,
) -> dict:
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

    log.info(f"Best: {best_bs}  cut={best_cut:.3f}  |  BF optimal: {bf_bs}  cut={bf_cut:.3f}")
    log.info(f"Approximation ratio: {ar:.4f}")

    top5 = sorted(corrected.items(), key=lambda x: -x[1])[:5]
    rows = "\n".join(f"| `{bs}` | {prob:.4f} |" for bs, prob in top5)
    hist_summary = (
        f"Start: {opt_result['energy_history'][0]:.4f} → "
        f"End: {opt_result['energy_history'][-1]:.4f}  "
        f"({opt_result['n_iter']} evaluations)"
    )

    create_markdown_artifact(
        key="qaoa-results",
        markdown=f"""## QAOA Results

| Metric | Value |
|--------|-------|
| **Best bitstring** | `{best_bs}` |
| **QAOA cut value** | {best_cut:.4f} |
| **Optimal cut (brute-force)** | {bf_cut:.4f} |
| **Approximation ratio** | {ar:.4f} |
| **COBYLA converged** | {opt_result['converged']} |
| **Iterations** | {opt_result['n_iter']} |

### Optimisation History
{hist_summary}

### Top-5 Bitstrings (REM-corrected)
| Bitstring | Probability |
|-----------|-------------|
{rows}
""",
        description="QAOA final results with REM correction",
    )

    return {
        "best_bitstring": best_bs,
        "best_cut_value": best_cut,
        "optimal_cut_value": bf_cut,
        "approximation_ratio": ar,
        "corrected_distribution": corrected,
        "optimization": opt_result,
    }


def _brute_force_maxcut(n: int, edges: list) -> tuple:
    best, best_bs = 0.0, "0" * n
    for bits in itertools.product([0, 1], repeat=n):
        cut = sum(w for u, v, w in edges if bits[u] != bits[v])
        if cut > best:
            best, best_bs = cut, "".join(map(str, bits))
    return best, best_bs


# ═══════════════════════════════════════════════════════════════════════
# MAIN FLOW
# ═══════════════════════════════════════════════════════════════════════

@flow(
    name="qaoa-iqm-resonance",
    description=(
        "QAOA on IQM Resonance — COBYLA optimisation + "
        "Readout Error Mitigation (pluggable via rem.py)"
    ),
    log_prints=True,
)
def qaoa_iqm_flow(
    n_qubits: int = 4,
    p_layers: int = 2,
    shots: int = 4096,
    cobyla_max_iter: int = 150,
    cobyla_rhobeg: float = 0.5,
    rem_shots: int = 1024,
    seed: int = 42,
):
    """
    QAOA pipeline stages:
      1. Configure IQM Resonance target (or local simulator if no token)
      2. Build MaxCut problem graph
      3. Build cost Hamiltonian
      4. Calibrate readout noise          ← rem.calibrate()
      5. Build QAOA kernel                ← CUDA-Q
      6. COBYLA optimisation loop         ← rem.apply() every iteration
      7. Post-process and report results
    """
    print(f"\n{'━' * 60}")
    print(f"  QAOA · IQM Resonance")
    print(f"  Qubits: {n_qubits}  |  QAOA depth p: {p_layers}")
    print(f"  Shots: {shots}  |  COBYLA max iter: {cobyla_max_iter}")
    print(f"  REM calibration shots: {rem_shots}  |  Seed: {seed}")
    print(f"{'━' * 60}\n")

    # Stage 1
    print("▸ STAGE 1: Configure IQM Target")
    configure_iqm_target()

    # Stage 2
    print("\n▸ STAGE 2: Build Problem Graph")
    graph = build_problem_graph(n_qubits, seed)

    # Stage 3
    print("\n▸ STAGE 3: Build Cost Hamiltonian")
    hamiltonian = build_cost_hamiltonian(graph)

    # Stage 4
    print("\n▸ STAGE 4: Calibrate Readout (rem.calibrate)")
    cal = calibrate_readout(n_qubits, rem_shots)

    # Stage 5 + 6
    print("\n▸ STAGE 5+6: QAOA Circuit + COBYLA Optimisation")
    opt_result = run_cobyla_optimization(
        graph=graph,
        cal=cal,
        p=p_layers,
        shots=shots,
        max_iter=cobyla_max_iter,
        rhobeg=cobyla_rhobeg,
    )

    # Stage 7
    print("\n▸ STAGE 7: Post-Process Results")
    final = post_process_results(
        graph=graph,
        opt_result=opt_result,
        cal=cal,
        shots=shots,
        p=p_layers,
    )

    print(f"\n{'━' * 60}")
    print(f"  Pipeline complete!")
    print(f"  Best bitstring     : {final['best_bitstring']}")
    print(f"  QAOA cut value     : {final['best_cut_value']:.4f}")
    print(f"  Optimal cut (BF)   : {final['optimal_cut_value']:.4f}")
    print(f"  Approximation ratio: {final['approximation_ratio']:.4f}")
    print(f"  COBYLA converged   : {final['optimization']['converged']}")
    print(f"  Iterations         : {final['optimization']['n_iter']}")
    print(f"  Artifacts          : Prefect dashboard → Artifacts tab")
    print(f"{'━' * 60}\n")

    return final


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QAOA on IQM Resonance · CUDA-Q · Prefect")
    parser.add_argument("--n_qubits",       type=int,   default=4)
    parser.add_argument("--p_layers",       type=int,   default=2)
    parser.add_argument("--shots",          type=int,   default=4096)
    parser.add_argument("--cobyla_max_iter",type=int,   default=150)
    parser.add_argument("--cobyla_rhobeg",  type=float, default=0.5)
    parser.add_argument("--rem_shots",      type=int,   default=1024)
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    qaoa_iqm_flow(
        n_qubits=args.n_qubits,
        p_layers=args.p_layers,
        shots=args.shots,
        cobyla_max_iter=args.cobyla_max_iter,
        cobyla_rhobeg=args.cobyla_rhobeg,
        rem_shots=args.rem_shots,
        seed=args.seed,
    )