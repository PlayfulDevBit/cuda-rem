QAOA on IQM Resonance — Qiskit + Prefect Pipeline
Runs the QAOA (Quantum Approximate Optimisation Algorithm) on IQM Resonance quantum hardware using Qiskit 2.1.2, with Readout Error Mitigation (REM) and automated orchestration via Prefect 3.x Cloud.

What the pipeline does
Connects to IQM Resonance cloud hardware via Qiskit IQM provider
Builds a MaxCut optimisation problem (swappable for your own)
Builds a parameterised QAOA circuit
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
Step 1 — Install dependencies
bash
pip install qiskit==2.1.2 qiskit-aer scipy numpy jupyter
Step 2 — Copy rem.py into your working directory
Copy rem.py from this repository into the same folder as your Jupyter notebook.

Step 3 — Open rem.py and edit your algorithm
The file has two clearly marked sections:

calibrate(backend, n_qubits, shots) — runs calibration circuits and returns calibration data
apply(raw_counts, calibration, n_qubits, shots) — takes raw counts and returns corrected probabilities
Replace the body of either function. The only rule is:

apply() must return a dict mapping bitstrings to probabilities. Example: {"00": 0.12, "01": 0.43, "10": 0.31, "11": 0.14}

Step 4 — Test in Jupyter
python
from qiskit_aer import AerSimulator

# Local simulator — no IQM hardware or token needed
backend = AerSimulator()

from rem import calibrate, apply

# Run your calibration
cal = calibrate(backend=backend, n_qubits=2, shots=1024)
print("Calibration data:", cal)

# Test correction on some raw counts
raw = {"00": 600, "01": 200, "10": 150, "11": 50}
corrected = apply(raw_counts=raw, calibration=cal, n_qubits=2, shots=1000)
print("Corrected probabilities:", corrected)
Step 5 — Validate end-to-end locally
bash
# Uses local Aer simulator if no IQM token is set
python qaoa_iqm_pipeline.py

# Run on real IQM Resonance hardware
export IQM_TOKEN="your-token-here"
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
The next Prefect Cloud run automatically uses your updated rem.py — no other files change
Pipeline owner: setup and deployment
Step 1 — Configure Prefect Cloud
bash
pip install "prefect>=3.0"
prefect cloud login
Step 2 — Store the IQM token as a Prefect Secret
python
from prefect.blocks.system import Secret
Secret(value="your-iqm-resonance-api-key").save("iqm-resonance-token")
Step 3 — Push code to GitHub
bash
git push origin main
Step 4 — Set your work pool name and register the deployment
Edit deploy_qaoa.py and set WORK_POOL_NAME to your Prefect Cloud work pool name, then:

bash
python deploy_qaoa.py
Step 5 — Trigger a run
From the CLI:

bash
prefect deployment run qaoa-iqm-resonance/qaoa-iqm-resonance
Or from the Prefect Cloud UI:

Go to Deployments → qaoa-iqm-resonance
Click Run → Custom Run
Override any parameters (n_qubits, shots, p_layers, etc.)
Click Submit
Monitor in Flow Runs → check Artifacts tab for results
Requirements
qiskit==2.1.2
iqm-client[qiskit]==33.0.5
qiskit-aer
prefect>=3.0
scipy
numpy>=1.24
networkx
How the files connect
deploy_qaoa.py  →  qaoa_iqm_pipeline.py  →  rem.py
deploy_qaoa.py registers the flow with Prefect Cloud pointing at the GitHub repo. qaoa_iqm_pipeline.py defines all pipeline stages and imports calibrate and apply directly from rem.py. When you update rem.py and push, the next run uses it automatically.

