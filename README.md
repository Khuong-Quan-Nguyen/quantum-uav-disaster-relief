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

**Option A — standalone script (original, no frontend):**
```bash
pip install -r requirements.txt
python uav_placement_qaoa.py
```

**Option B — with the web dashboard:**
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```
Then open `http://localhost:8000` in a browser. Click **"Run optimization"**
to dispatch the real QAOA pipeline (~2-3s) and watch the live optimizer log,
mission map, and classical-vs-quantum comparison update.

The dashboard is a dark ops-console UI: a live terminal-style log streams
the optimizer's actual progress (not a fake loading bar), and the map lights
up UAV-to-zone connections once an assignment resolves.

## Possible extensions

- Scale up to more UAVs/zones (note: qubit count grows as UAVs × zones,
  so this needs either more qubits or a smarter encoding).
- Add real-time sensor inputs (water level, wind) as early-warning
  triggers for re-optimizing UAV assignment.
- Compare QAOA against classical heuristics (K-means + greedy) instead of
  brute force, once the problem is too large to solve exactly.
