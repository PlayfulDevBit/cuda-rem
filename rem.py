"""
rem.py — Readout Error Mitigation
===================================
THIS IS THE FILE YOU EDIT.

There are exactly two functions you need to implement:
    - calibrate()  : runs circuits to characterise the device readout noise
    - apply()      : uses the calibration data to correct raw measurement counts

The rest of the pipeline (qaoa_iqm_pipeline.py) imports these two functions
and calls them automatically. You never need to touch the pipeline file.

CONTRACT (do not change function names or argument names):
    calibrate(n_qubits, shots)                              → any object you choose
    apply(raw_counts, calibration, n_qubits, shots)         → dict[str, float]

The return of apply() must be a dict mapping bitstrings to corrected probabilities.
Example: {"00": 0.12, "01": 0.43, "10": 0.31, "11": 0.14}

DEVELOPING IN JUPYTER:
    import cudaq
    import os
    os.environ["CUDAQ_TARGET"] = "qpp-cpu"   # local simulator — no hardware needed
    cudaq.set_target("qpp-cpu")

    from rem import calibrate, apply
    cal = calibrate(n_qubits=2, shots=1024)
    corrected = apply(raw_counts={"00": 600, "01": 400}, calibration=cal, n_qubits=2, shots=1000)
    print(corrected)
"""

import numpy as np
from scipy.optimize import lsq_linear
import cudaq


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# Feel free to replace the body of this function entirely.
# Just make sure it returns something your apply() function understands.
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(n_qubits: int, shots: int) -> np.ndarray:
    """
    Run calibration circuits on the current CUDA-Q target and return
    calibration data that apply() will use to correct raw counts.

    DEFAULT IMPLEMENTATION:
        Builds the full 2^n x 2^n assignment matrix A where:
            A[i, j] = P(measuring bitstring i | device was prepared in state j)

    PARAMETERS:
        n_qubits : number of qubits in your circuit
        shots    : number of measurement shots per calibration circuit

    RETURNS:
        np.ndarray of shape (2^n_qubits, 2^n_qubits)
        (replace with any structure your apply() expects)

    ── EDIT BELOW THIS LINE ─────────────────────────────────────────────────
    """
    dim = 2 ** n_qubits
    A = np.zeros((dim, dim))

    for j in range(dim):
        bits = [int(b) for b in format(j, f"0{n_qubits}b")]

        @cudaq.kernel
        def cal_circuit(target_bits: list[int]):
            q = cudaq.qvector(n_qubits)
            for idx, bit in enumerate(target_bits):
                if bit == 1:
                    x(q[idx])

        counts = cudaq.sample(cal_circuit, bits, shots_count=shots)
        for meas, cnt in counts.items():
            i = int(meas, 2)
            A[i, j] = cnt / shots

    return A
    # ── EDIT ABOVE THIS LINE ─────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTION
# Feel free to replace the body of this function entirely.
# Just make sure it returns a dict of {bitstring: probability}.
# ─────────────────────────────────────────────────────────────────────────────

def apply(
    raw_counts: dict,
    calibration: np.ndarray,
    n_qubits: int,
    shots: int,
) -> dict:
    """
    Apply readout error mitigation to raw measurement counts.

    DEFAULT IMPLEMENTATION:
        Solves  A @ p_ideal = p_noisy  via constrained least-squares,
        then renormalises so probabilities sum to 1.

    PARAMETERS:
        raw_counts  : dict mapping bitstrings to integer counts
                      e.g. {"00": 612, "01": 388, "10": 0, "11": 0}
        calibration : whatever calibrate() returned
        n_qubits    : number of qubits
        shots       : total number of shots (used to normalise raw_counts)

    RETURNS:
        dict mapping bitstrings to corrected float probabilities
        e.g. {"00": 0.63, "01": 0.37, "10": 0.0, "11": 0.0}

    ── EDIT BELOW THIS LINE ─────────────────────────────────────────────────
    """
    dim = 2 ** n_qubits
    A = calibration

    # Build noisy probability vector
    p_noisy = np.zeros(dim)
    for bs, cnt in raw_counts.items():
        p_noisy[int(bs, 2)] = cnt / shots

    # Constrained least-squares: p >= 0
    result = lsq_linear(A, p_noisy, bounds=(0, 1))
    p_ideal = result.x
    total = p_ideal.sum()
    if total > 0:
        p_ideal /= total  # renormalise

    return {format(i, f"0{n_qubits}b"): float(p_ideal[i]) for i in range(dim)}
    # ── EDIT ABOVE THIS LINE ─────────────────────────────────────────────────