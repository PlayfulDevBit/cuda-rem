"""
run.py — Run the QAOA + REM pipeline
======================================
This is the only file you need to run, whether you are:
    - A researcher validating a new REM algorithm locally
    - A pipeline owner triggering a production run on IQM Resonance

USAGE:
    python run.py                        # uses DEFAULT_CONFIG below
    python run.py --shots 8192           # override any config value
    prefect run run.py                   # trigger via Prefect (production)

ENVIRONMENT VARIABLES (required for real hardware):
    IQM_TOKEN       – your IQM Resonance API token
    IQM_SERVER_URL  – e.g. https://resonance.meetiqm.com/cocos

    For local testing without hardware, set:
    CUDAQ_TARGET=qpp-cpu               – use the local CPU simulator instead
"""

import argparse
import json
import os
from qaoa_iqm_pipeline import qaoa_iqm_flow

# ─────────────────────────────────────────────
# CONFIGURATION  — edit these defaults freely
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    "n_qubits": 4,           # number of qubits
    "p_layers": 2,           # QAOA depth
    "shots": 4096,           # shots per circuit execution
    "cobyla_max_iter": 150,  # COBYLA iteration budget
    "cobyla_rhobeg": 0.5,    # COBYLA initial simplex size
    "rem_shots": 1024,       # shots per calibration circuit
    "seed": 42,              # random seed for reproducibility
}


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run QAOA on IQM Resonance")
    for key, val in DEFAULT_CONFIG.items():
        parser.add_argument(f"--{key}", type=type(val), default=val)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)

    # Use local simulator if IQM credentials are not set
    if not os.environ.get("IQM_TOKEN"):
        print("⚠️  IQM_TOKEN not set — falling back to local CPU simulator.")
        os.environ["CUDAQ_TARGET"] = "qpp-cpu"

    print(f"\nRunning with config:\n{json.dumps(config, indent=2)}\n")

    result = qaoa_iqm_flow(config=config)

    print("\n── Results ──────────────────────────────")
    print(f"Best bitstring     : {result['best_bitstring']}")
    print(f"QAOA cut value     : {result['best_cut_value']:.4f}")
    print(f"Optimal cut (BF)   : {result['optimal_cut_value']:.4f}")
    print(f"Approximation ratio: {result['approximation_ratio']:.4f}")
    print(f"COBYLA converged   : {result['optimization']['converged']}")
    print(f"Iterations         : {result['optimization']['n_iter']}")
