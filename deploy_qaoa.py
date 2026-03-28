"""
deploy_qaoa.py — Prefect 3.x Cloud Deployment
===============================================
Registers the QAOA + REM pipeline to Prefect Cloud.

BEFORE RUNNING:
  1. Set WORK_POOL_NAME below to your Prefect Cloud work pool name.
     Find it at: Prefect Cloud UI → Work Pools
     If you don't have one, create a managed process pool in the UI first.

  2. Push the pipeline code to GitHub:
       git push origin main

  3. Ensure your IQM token is stored as a Prefect Secret:
       from prefect.blocks.system import Secret
       Secret(value="your-iqm-key").save("iqm-resonance-token")

Usage:
    python deploy_qaoa.py
"""

import sys
import unittest.mock

# Qiskit IQM provider is Linux-only — mock it so this deploy script
# can run on Windows without installing the full quantum stack
for mod in [
    "qiskit_aer",
    "iqm",
    "iqm.qiskit_iqm",
]:
    sys.modules.setdefault(mod, unittest.mock.MagicMock())

from prefect.runner.storage import GitRepository
from qaoa_iqm_pipeline import qaoa_iqm_flow

# ── CONFIGURE THIS ──────────────────────────────────────────────────────
WORK_POOL_NAME = "my-managed-pool"   # ← replace with your work pool name
# ────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    deployment = qaoa_iqm_flow.from_source(
        source=GitRepository(
            url="https://github.com/PlayfulDevBit/cuda-rem.git",
            branch="main",
        ),
        entrypoint="qaoa_iqm_pipeline.py:qaoa_iqm_flow",
    )

    deployment_id = deployment.deploy(
        name="qaoa-iqm-resonance",
        work_pool_name=WORK_POOL_NAME,
        version="1.0.0",
        description=(
            "QAOA on IQM Resonance via Qiskit. "
            "COBYLA optimisation with pluggable Readout Error Mitigation (rem.py). "
            "Produces cost history and results artifacts."
        ),
        tags=["quantum", "qaoa", "qiskit", "iqm-resonance", "rem", "cobyla"],

        # ── Default Parameters (all overridable from Prefect Cloud UI) ──
        parameters={
            "n_qubits":        4,
            "p_layers":        2,
            "shots":           4096,
            "cobyla_max_iter": 150,
            "cobyla_rhobeg":   0.5,
            "rem_shots":       1024,
            "seed":            42,
        },

        # ── Dependencies — pinned to match working Grover pipeline ──────
        job_variables={
            "pip_packages": [
                "qiskit==2.1.2",
                "iqm-client[qiskit]==33.0.5",
                "qiskit-aer",
                "prefect>=3.0",
                "scipy",
                "numpy>=1.24",
                "networkx",
            ],
        },
    )

    print(f"\nDeployment registered: {deployment_id}")
    print("\nTrigger with:")
    print("  prefect deployment run qaoa-iqm-resonance/qaoa-iqm-resonance")
    print("\nParameter overrides available in Prefect Cloud UI:")
    print("  n_qubits         int    — number of qubits")
    print("  p_layers         int    — QAOA circuit depth")
    print("  shots            int    — shots per circuit execution")
    print("  cobyla_max_iter  int    — COBYLA iteration budget")
    print("  cobyla_rhobeg    float  — COBYLA initial simplex size")
    print("  rem_shots        int    — shots per REM calibration circuit")
    print("  seed             int    — random seed for problem graph")
    print("\nPrerequisites:")
    print("  - Prefect Cloud account with PREFECT_API_URL and PREFECT_API_KEY set")
    print("  - Prefect Secret block 'iqm-resonance-token' containing IQM Resonance API key")
    print("  - Code pushed to: https://github.com/PlayfulDevBit/cuda-rem.git")