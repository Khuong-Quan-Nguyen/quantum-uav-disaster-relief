# Quantum-Optimized UAV Placement for Post-Flood Network Restoration

A small, honest demo of **quantum optimization (QAOA)** applied to a
disaster-relief communications problem: after a flood destroys ground
telecom infrastructure, which UAV (flying temporary base station) should
cover which affected zone, to restore connectivity as fast as possible?

## Why this project

Central Vietnam is hit by severe flooding most years, and floods are a
leading cause of Vietnam's roughly 10,000 annual disaster-related deaths.
Restoring communication quickly after a disaster is directly tied to
faster rescue coordination. This mirrors real published research on
UAV-aided disaster-relief networks (clustering affected users, then
optimally allocating UAVs/resources to serve them) — an active area
combining wireless communications and optimization, including quantum
optimization.

This demo is illustrative, not a live system: the flood zone coordinates
are approximate real locations (Quảng Bình, Quảng Trị, Thừa Thiên Huế), but
there's no live sensor or disaster data feeding it.

## How it works

1. **Scenario**: a small set of flood-affected provinces and UAV staging
   points, converted to great-circle distances.
2. **Classical baseline**: brute-force search for the assignment (which UAV
   covers which zone) that minimizes total distance — the true optimum for
   this small problem size.
3. **Quantum approach**: the same assignment problem is formulated as a
   **QUBO** (Quadratic Unconstrained Binary Optimization) — one qubit per
   (UAV, zone) pair, with a penalty term enforcing "each UAV covers exactly
   one zone." The QUBO is converted to an Ising Hamiltonian and solved with
   **QAOA** (Quantum Approximate Optimization Algorithm) via PennyLane,
   trained with multiple random restarts (a standard, honest way to handle
   QAOA's sensitivity to initialization on a small demo).
4. The QUBO-to-Hamiltonian conversion is verified against the direct cost
   function for **every possible bitstring** before being trusted — worth
   keeping in any real use of this pattern, since the conversion is easy to
   get subtly wrong.
5. Output: printed comparison of both assignments + a map-style plot.

## Honest result

On this small problem, QAOA converges to the **same optimal assignment**
as the classical brute-force search. That's expected and appropriate to
state plainly: for a problem this size, classical brute force is trivial
and QAOA isn't demonstrating a speed advantage — it's demonstrating a
correct, working, from-scratch implementation of the QUBO formulation and
quantum optimization pipeline that these problems are built on. Quantum
advantage (if any) would only show up at a much larger scale than a laptop
simulator can run.

## Run it

**Live demo:** _(add your Vercel URL here after the first deploy)_

**Option A — in the browser, nothing to install:**
Open `frontend/index.html` directly, or serve the folder:
```bash
python -m http.server -d frontend 8000   # then open http://localhost:8000
```
The entire pipeline — QUBO construction, the all-bitstrings verification,
QAOA training over 3 restarts, and readout — runs client-side in
[`frontend/qaoa.js`](frontend/qaoa.js) in well under a second. No Python, no
backend, no PennyLane. **This is what the deployed site runs.**

**Option B — PennyLane reference implementation (standalone script):**
```bash
pip install -r requirements.txt
python uav_placement_qaoa.py
```
Prints both assignments and saves a map plot to `uav_assignment.png`.

**Option C — FastAPI backend serving the same dashboard:**
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```
Kept as the reference/server-side path. Note the deployed site does **not**
use this — see below.

The dashboard is a dark ops-console UI: a live terminal-style log streams the
optimizer's actual progress (not a fake loading bar), and the map lights up
UAV-to-zone connections once an assignment resolves.

## Two implementations, cross-checked

The physics is implemented twice, on purpose:

| | Implementation | Role |
|---|---|---|
| Reference | [`backend/optimizer.py`](backend/optimizer.py) | PennyLane + `lightning.qubit`. The thing that gets to be right. |
| Deployed | [`frontend/qaoa.js`](frontend/qaoa.js) | From-scratch complex statevector simulator — Hadamard init, diagonal cost-layer phases, strided RX mixer butterflies, hand-rolled Adam matching PennyLane's defaults. |

That claim is checkable rather than asserted.
[`tools/crosscheck.py`](tools/crosscheck.py) runs both over identical fixed
parameters and compares the QUBO matrices, the Ising Hamiltonian, the full
64-amplitude probability distribution, and the energy expectation. The JS side
is executed by headless Chrome via `--dump-dom`, so no Node install is needed:

```bash
backend/venv/Scripts/python.exe tools/crosscheck.py
```

Both implementations also verify the QUBO→Ising conversion against the direct
cost function for **all 2⁶ bitstrings** at runtime before trusting it — that
conversion is exactly where these implementations tend to break silently.

## Deploying

The site is pure static. [`vercel.json`](vercel.json) points Vercel at
`frontend/` as the output directory; there is no build step and no serverless
function. The Python backend stays in the repo as the reference
implementation, but nothing on the deployed page depends on it.

## Possible extensions

- Scale up to more UAVs/zones (note: qubit count grows as UAVs × zones,
  so this needs either more qubits or a smarter encoding).
- Add real-time sensor inputs (water level, wind) as early-warning
  triggers for re-optimizing UAV assignment.
- Compare QAOA against classical heuristics (K-means + greedy) instead of
  brute force, once the problem is too large to solve exactly.
