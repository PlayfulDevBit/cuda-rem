# QAOA on IQM Resonance — Prefect Pipeline

A research pipeline that runs the **QAOA** (Quantum Approximate Optimisation Algorithm) on **IQM Resonance** quantum hardware, with **Readout Error Mitigation (REM)** and automated orchestration via **Prefect**.

---

## What the pipeline does

1. Connects to IQM Resonance cloud hardware via CUDA-Q
2. Builds a MaxCut optimisation problem (swappable for your own)
3. Calibrates the device's readout noise
4. Runs a QAOA variational loop, optimised with COBYLA
5. Applies readout error mitigation to every circuit result
6. Reports the best solution found and the approximation ratio

---

## File overview

| File | What it is | Who touches it |
|------|-----------|----------------|
| `rem.py` | Your REM algorithm — `calibrate()` and `apply()` | **Researcher** |
| `run.py` | Single entry point to run everything | Researcher + pipeline owner |
| `qaoa_iqm_pipeline.py` | Prefect tasks and flow logic | Pipeline owner |
| `deploy.py` | Prefect deployment, scheduling, Docker config | Pipeline owner / DevOps |
| `README.md` | This file | Everyone |

---

## Researcher guide: developing your own REM in Jupyter

This section is for you if you want to implement a custom readout error mitigation algorithm and test it without worrying about the pipeline.

### Step 1 — Set up your environment

```bash
pip install cuda-quantum scipy numpy jupyter
```

Set your IQM credentials (or skip for local simulation):

```bash
export IQM_TOKEN="your-token-here"
export IQM_SERVER_URL="https://resonance.meetiqm.com/cocos"
```

To use the local CPU simulator instead of real hardware (no credentials needed):

```bash
export CUDAQ_TARGET=qpp-cpu
```

### Step 2 — Copy rem.py into your working directory

Copy `rem.py` from this repository into the same folder as your notebook.

### Step 3 — Open rem.py and edit your algorithm

The file has two clearly marked sections:

- `calibrate(n_qubits, shots)` — runs circuits to measure the device's readout noise and returns calibration data
- `apply(raw_counts, calibration, n_qubits, shots)` — takes raw circuit results and returns corrected probabilities

Replace the body of either function with your own algorithm. The only rule is:

> `apply()` must return a `dict` mapping bitstrings to probabilities.  
> Example: `{"00": 0.12, "01": 0.43, "10": 0.31, "11": 0.14}`

### Step 4 — Test in Jupyter

```python
import cudaq
import os

# Use local simulator for testing
os.environ["CUDAQ_TARGET"] = "qpp-cpu"
cudaq.set_target("qpp-cpu")

from rem import calibrate, apply

# Run calibration
cal = calibrate(n_qubits=2, shots=1024)
print("Calibration data:", cal)

# Test correction on some fake raw counts
raw = {"00": 600, "01": 200, "10": 150, "11": 50}
corrected = apply(raw_counts=raw, calibration=cal, n_qubits=2, shots=1000)
print("Corrected probabilities:", corrected)
```

Iterate freely. Run cells, change your algorithm, re-run. No pipeline involved.

### Step 5 — Validate end-to-end with run.py

Once your REM looks good in Jupyter, copy your updated `rem.py` into the pipeline folder and run:

```bash
python run.py
```

This runs the full QAOA pipeline — problem construction, calibration, optimisation loop, post-processing — using your REM algorithm. Results are printed to the terminal.

To test with the local simulator first:

```bash
python run.py   # automatically falls back to simulator if IQM_TOKEN is not set
```

To run on real IQM Resonance hardware:

```bash
export IQM_TOKEN="your-token-here"
export IQM_SERVER_URL="https://resonance.meetiqm.com/cocos"
python run.py
```

---

## Injecting your REM back into the pipeline

Once you are happy with `run.py` results, handing your REM back to the pipeline is a single step:

1. Copy your updated `rem.py` into the pipeline repository
2. Tell the pipeline owner — they will verify the two function signatures are intact and trigger a production run via Prefect

That is all. You do not need to touch `qaoa_iqm_pipeline.py` or `deploy.py`.

---

## Pipeline owner: triggering a production run

Register the deployment (first time or after config changes):

```bash
python deploy.py
```

Trigger a run from the CLI:

```bash
prefect deployment run qaoa-iqm-resonance/iqm-resonance-deploy
```

Or trigger from the Prefect UI with custom parameters.

---

## Requirements

```
cuda-quantum
prefect>=2.0
scipy
numpy
networkx
```

Install with:

```bash
pip install cuda-quantum prefect scipy numpy networkx
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `IQM_TOKEN` | Yes (hardware) | IQM Resonance API token |
| `IQM_SERVER_URL` | Yes (hardware) | IQM Resonance server URL |
| `CUDAQ_TARGET` | No | Set to `qpp-cpu` to use local simulator |
