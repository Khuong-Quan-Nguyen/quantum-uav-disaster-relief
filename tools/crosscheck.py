"""
Cross-check the browser QAOA simulator (frontend/qaoa.js) against the
PennyLane reference implementation (backend/optimizer.py).

The browser sim is what the deployed site runs; PennyLane is the thing that
gets to be right. This script makes the claim checkable rather than asserted:
it runs both over identical fixed parameters and compares the QUBO matrices,
the Ising Hamiltonian, the full statevector probability distribution, and the
energy expectation.

The JS side is executed by headless Chrome (no Node required): the harness at
tools/crosscheck.html computes the fixtures and writes them into the DOM,
which `chrome --dump-dom` hands back to us.

Usage:
    backend/venv/Scripts/python.exe tools/crosscheck.py
"""

import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import qaoa

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from optimizer import FLOOD_ZONES, UAV_BASES, build_distance_matrix  # noqa: E402

PENALTY = 1.5
N_LAYERS = 2
TOL = 1e-9

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_browser():
    for c in CHROME_CANDIDATES:
        if os.path.exists(c):
            return c
    for name in ("google-chrome", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("No Chrome/Chromium/Edge found -- needed to execute the JS side.")


def run_js():
    """Execute tools/crosscheck.html in headless Chrome and parse the JSON it emits."""
    browser = find_browser()
    harness = (ROOT / "tools" / "crosscheck.html").as_uri()
    with tempfile.TemporaryDirectory() as profile:
        proc = subprocess.run(
            [
                browser, "--headless=new", "--disable-gpu", "--no-sandbox",
                "--allow-file-access-from-files",
                "--virtual-time-budget=60000",
                f"--user-data-dir={profile}",
                "--dump-dom", harness,
            ],
            capture_output=True, text=True, timeout=180,
        )
    m = re.search(r'<pre id="out">(.*?)</pre>', proc.stdout, re.S)
    if not m:
        raise SystemExit(f"Could not read harness output.\nstdout head:\n{proc.stdout[:2000]}")
    raw = html.unescape(m.group(1)).strip()
    if raw == "PENDING":
        raise SystemExit("Harness did not finish -- raise --virtual-time-budget.")
    return json.loads(raw)


def build_python_side():
    """The same QUBO -> Ising construction as backend/optimizer.py, exposed for comparison."""
    distance, uav_names, zone_names = build_distance_matrix(FLOOD_ZONES, UAV_BASES)
    n_uavs, n_zones = len(uav_names), len(zone_names)
    n_qubits = n_uavs * n_zones

    def idx(u, z):
        return u * n_zones + z

    distance_norm = distance / distance.max()

    diag = np.zeros(n_qubits)
    pair = np.zeros((n_qubits, n_qubits))
    const = 0.0
    for u in range(n_uavs):
        for z in range(n_zones):
            diag[idx(u, z)] += distance_norm[u, z]
    for u in range(n_uavs):
        ids = [idx(u, z) for z in range(n_zones)]
        for i in ids:
            diag[i] += -PENALTY
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pair[ids[a], ids[b]] += 2 * PENALTY
        const += PENALTY

    coeffs, ops = [], []
    h = np.zeros(n_qubits)
    for i in range(n_qubits):
        neighbor_sum = sum(pair[i, j] for j in range(n_qubits) if j > i) + \
                       sum(pair[j, i] for j in range(n_qubits) if j < i)
        h[i] = -diag[i] / 2 - neighbor_sum / 4
        coeffs.append(h[i])
        ops.append(qml.PauliZ(i))
    J = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if pair[i, j] != 0:
                J[i, j] = pair[i, j] / 4
                coeffs.append(J[i, j])
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    cost_h = qml.Hamiltonian(coeffs, ops)
    offset = diag.sum() / 2 + pair.sum() / 4

    return dict(
        distance=distance, n_qubits=n_qubits, diag=diag, pair=pair, const=const,
        h=h, J=J, offset=offset, cost_h=cost_h,
    )


def make_qnodes(cost_h, n_qubits):
    mixer_h = qaoa.x_mixer(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    def ansatz(gammas, betas):
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        for l in range(len(gammas)):
            qaoa.cost_layer(gammas[l], cost_h)
            qaoa.mixer_layer(betas[l], mixer_h)

    @qml.qnode(dev)
    def q_expval(gammas, betas):
        ansatz(gammas, betas)
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def q_probs(gammas, betas):
        ansatz(gammas, betas)
        return qml.probs(wires=range(n_qubits))

    return q_expval, q_probs


def check(label, a, b, tol=TOL):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        print(f"  FAIL  {label}: shape {a.shape} vs {b.shape}")
        return False, np.inf
    err = float(np.max(np.abs(a - b))) if a.size else 0.0
    ok = err <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<44} max|diff| = {err:.3e}")
    return ok, err


def main():
    print("Running JS simulator in headless Chrome ...")
    js = run_js()
    print("Building PennyLane reference ...")
    py = build_python_side()
    n = py["n_qubits"]
    q_expval, q_probs = make_qnodes(py["cost_h"], n)

    results = []
    print("\n--- scenario & QUBO construction ---")
    results.append(check("distance matrix (km)", js["distance_matrix"], py["distance"], 1e-9))
    results.append(check("QUBO linear terms", js["qubo"]["diag"], py["diag"]))
    results.append(check("QUBO quadratic terms",
                         np.array(js["qubo"]["pair"]).reshape(n, n), py["pair"]))
    results.append(check("QUBO constant", [js["qubo"]["const_term"]], [py["const"]]))

    print("\n--- QUBO -> Ising conversion ---")
    results.append(check("Ising fields h_i", js["ising"]["h"], py["h"]))
    results.append(check("Ising couplings J_ij",
                         np.array(js["ising"]["J"]).reshape(n, n), py["J"]))
    results.append(check("Ising constant offset", [js["ising"]["offset"]], [py["offset"]]))

    print("\n--- statevector simulation vs PennyLane default.qubit ---")
    for k, fx in enumerate(js["fixtures"]):
        g = np.array(fx["params"]["gammas"], dtype=float)
        b = np.array(fx["params"]["betas"], dtype=float)
        results.append(check(f"fixture {k}: probs (64 amplitudes)",
                             fx["probs"], np.array(q_probs(g, b), dtype=float)))
        results.append(check(f"fixture {k}: <H_C>",
                             [fx["expval"]], [float(q_expval(g, b))]))

    print("\n--- end-to-end pipeline (seeded JS run) ---")
    pipeline = js["pipeline"]
    print(f"  JS verification max error over all 2^{n} bitstrings: "
          f"{pipeline['verification_max_error']:.3e}")
    print(f"  classical: {pipeline['classical']['assignment']}  "
          f"{pipeline['classical']['total_km']:.2f} km")
    print(f"  quantum:   {pipeline['quantum']['assignment']}  "
          f"{pipeline['quantum']['total_km']:.2f} km  valid={pipeline['quantum']['valid']}")
    agree = pipeline["quantum"]["assignment"] == pipeline["classical"]["assignment"]
    print(f"  {'PASS' if agree else 'NOTE'}  QAOA assignment matches brute-force optimum: {agree}")
    results.append((agree, 0.0))

    ok = all(r[0] for r in results)
    worst = max((r[1] for r in results if np.isfinite(r[1])), default=0.0)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'} "
          f"({len(results)} checks, worst deviation {worst:.3e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
