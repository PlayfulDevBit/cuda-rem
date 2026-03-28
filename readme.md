QAOA on IQM Resonance — CUDA-Q + Prefect Pipeline
Runs the QAOA (Quantum Approximate Optimisation Algorithm) on IQM Resonance quantum hardware using CUDA-Q, with Readout Error Mitigation (REM) and automated orchestration via Prefect 3.x Cloud.

What the pipeline does
Connects to IQM Resonance cloud hardware via CUDA-Q
Builds a MaxCut optimisation problem (swappable for your own)
Calibrates the device's readout noise (rem.calibrate)
Runs a QAOA variational loop optimised with COBYLA, applying REM every iteration (rem.apply)
Runs a final high-shot execution at optimal parameters
Reports the best solution found and the approximation ratio as Prefect artifacts
File overview
File	What it is	Who touches it
rem.py	Your REM algorithm — calibrate() and apply()	Researcher
qaoa_iqm_pipeline.py	Prefect flow, tasks, and CLI entry point	Pipeline owner
deploy_qaoa.py	Prefect Cloud deployment registration	Pipeline owner / DevOps
README.md	This file	Everyone
Researcher guide: developing your own REM in Jupyter
This section is for you if you want to implement a custom REM algorithm and test it without
worrying about the pipeline infrastructure.

Step 1 — Install dependencies
bash
pip install cuda-quantum scipy numpy jupyter
Step 2 — Copy rem.py into your working directory
Copy rem.py from this repository into the same folder as your Jupyter notebook.

Step 3 — Open rem.py and edit your algorithm
The file has two clearly marked sections:

calibrate(n_qubits, shots) — runs circuits to measure the device's readout noise
apply(raw_counts, calibration, n_qubits, shots) — takes raw counts and returns corrected probabilities
Replace the body of either function. The only rule is:

apply() must return a dict mapping bitstrings to probabilities. Example: {"00": 0.12, "01": 0.43, "10": 0.31, "11": 0.14}

Step 4 — Test in Jupyter
python
import cudaq
import os

# Use local CPU simulator — no IQM hardware or token needed
os.environ["CUDAQ_TARGET"] = "qpp-cpu"
cudaq.set_target("qpp-cpu")

from rem import calibrate, apply

# Run your calibration
cal = calibrate(n_qubits=2, shots=1024)
print("Calibration data:", cal)

# Test correction on some raw counts
raw = {"00": 600, "01": 200, "10": 150, "11": 50}
corrected = apply(raw_counts=raw, calibration=cal, n_qubits=2, shots=1000)
print("Corrected probabilities:", corrected)
Iterate freely — run cells, change your algorithm, re-run. The pipeline is not involved at all.

Step 5 — Validate end-to-end locally
Once your REM looks good in Jupyter, copy your updated rem.py into the repo folder and run the pipeline directly from the command line:

bash
# Uses local CPU simulator if no IQM token is set
python qaoa_iqm_pipeline.py

# Run on real IQM Resonance hardware
export IQM_TOKEN="your-token-here"
export IQM_SERVER_URL="https://cocos.resonance.meetiqm.com/garnet"
python qaoa_iqm_pipeline.py --n_qubits 4 --p_layers 2 --shots 4096
Available CLI flags:

--n_qubits         int    Number of qubits (default: 4)
--p_layers         int    QAOA circuit depth (default: 2)
--shots            int    Shots per execution (default: 4096)
--cobyla_max_iter  int    COBYLA iteration budget (default: 150)
--cobyla_rhobeg    float  COBYLA initial simplex size (default: 0.5)
--rem_shots        int    Shots per REM calibration circuit (default: 1024)
--seed             int    Random seed for problem graph (default: 42)
Injecting your REM back into the pipeline
Once you are happy with local results:

Copy your updated rem.py into the repository
Push to GitHub:
bash
   git add rem.py
   git commit -m "Update REM algorithm"
   git push origin main
The next Prefect Cloud run will automatically use your updated rem.py — no other files need to change
That is all. You do not need to touch qaoa_iqm_pipeline.py or deploy_qaoa.py.

Pipeline owner: setup and deployment
Step 1 — Configure Prefect Cloud
bash
pip install prefect>=3.0
prefect cloud login
Step 2 — Store the IQM token as a Prefect Secret
python
from prefect.blocks.system import Secret
Secret(value="your-iqm-resonance-api-key").save("iqm-resonance-token")
Step 3 — Push code to GitHub
bash
git push origin main
Step 4 — Set your work pool name and register the deployment
Edit deploy_qaoa.py and set WORK_POOL_NAME to your Prefect Cloud work pool name, then run:

bash
python deploy_qaoa.py
Step 5 — Trigger a run
From the CLI:

bash
prefect deployment run qaoa-iqm-resonance/qaoa-iqm-resonance
Or from the Prefect Cloud UI:

Go to Deployments → qaoa-iqm-resonance
Click Run → Custom Run
Override any parameters in the panel (n_qubits, shots, p_layers, etc.)
Click Submit
Monitor in Flow Runs → check the Artifacts tab for charts and results
Requirements
cuda-quantum
prefect>=3.0
scipy
numpy>=1.24
networkx
Install with:

bash
pip install cuda-quantum "prefect>=3.0" scipy "numpy>=1.24" networkx
Environment variables
Variable	Required	Description
IQM_TOKEN	Only for local runs without Prefect Secret	IQM Resonance API token
IQM_SERVER_URL	No	Defaults to https://cocos.resonance.meetiqm.com/garnet
In Prefect Cloud runs, the IQM token is loaded automatically from the iqm-resonance-token Secret block. No environment variables needed.

How the files connect
deploy_qaoa.py  →  qaoa_iqm_pipeline.py  →  rem.py
deploy_qaoa.py registers the flow with Prefect Cloud and points it at the GitHub repo. qaoa_iqm_pipeline.py defines all pipeline stages and imports calibrate and apply directly from rem.py. When you update rem.py, the pipeline automatically uses the new algorithm on the next run — no other files need to change.

