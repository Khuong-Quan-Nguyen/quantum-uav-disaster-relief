"""
Quantum-Optimized UAV Placement for Post-Flood Network Restoration
====================================================================

WHAT THIS DOES (plain English):
After a major flood, ground telecom infrastructure (cell towers, fiber) is
often destroyed or knocked offline. A fast way to restore connectivity for
rescue coordination is to fly UAVs (drones) that act as temporary flying
base stations, each one covering a cluster of affected communities.

The core engineering problem: which UAV should cover which flood-affected
zone, so that total UAV travel distance (and therefore time-to-connectivity)
is minimized? This is a classic *assignment problem*.

This script formulates that assignment as a QUBO (Quadratic Unconstrained
Binary Optimization) problem -- the standard way real-world optimization
problems get translated into a form a quantum computer (or quantum
simulator) can work on -- and solves it with QAOA (Quantum Approximate
Optimization Algorithm) using PennyLane. It's benchmarked against a
classical brute-force search on the same small problem.

WHY THIS PROBLEM: it mirrors real published research -- UAV-aided disaster
relief networks that use clustering + resource allocation to restore
connectivity after events like floods, which kill roughly 10,000 people a
year in Vietnam according to disaster-management reporting. This demo uses
illustrative coordinates for flood-prone Vietnamese provinces to keep the
scenario concrete and honest, not real live disaster data.

NO PRIOR QUANTUM COMPUTING KNOWLEDGE ASSUMED -- every quantum-specific line
is commented in plain language.
"""

import itertools
import numpy as np
import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as pnp  # PennyLane's differentiable numpy wrapper
import matplotlib.pyplot as plt

np.random.seed(0)
pnp.random.seed(0)

# ---------------------------------------------------------------------------
# 1. SCENARIO -- FLOOD-AFFECTED ZONES AND AVAILABLE UAVS
# ---------------------------------------------------------------------------
# Approximate lat/lon of provinces frequently hit by severe flooding in
# central Vietnam. Illustrative scenario, not live disaster data.
flood_zones = {
    "Quang Binh":      (17.47, 106.62),
    "Quang Tri":       (16.75, 107.19),
    "Thua Thien Hue":  (16.47, 107.59),
}

# UAV launch/staging points -- e.g. nearest operational airbase or relief
# staging area for each drone.
uav_bases = {
    "UAV-1": (17.85, 106.35),
    "UAV-2": (16.30, 107.70),
}

zone_names = list(flood_zones.keys())
uav_names = list(uav_bases.keys())
n_zones = len(zone_names)
n_uavs = len(uav_names)

def haversine_km(p1, p2):
    """Great-circle distance in km between two (lat, lon) points."""
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

# distance[i][j] = km from UAV i's base to flood zone j
distance = np.array([
    [haversine_km(uav_bases[u], flood_zones[z]) for z in zone_names]
    for u in uav_names
])

print("Distance matrix (km), rows=UAVs, cols=flood zones:")
print(np.round(distance, 1))

# ---------------------------------------------------------------------------
# 2. CLASSICAL BASELINE -- BRUTE-FORCE OPTIMAL ASSIGNMENT
# ---------------------------------------------------------------------------
# With only 3 zones and 2 UAVs (one zone goes uncovered on this pass -- a
# realistic constraint when drones are scarce right after a disaster), we
# can just try every possible assignment and pick the cheapest. This gives
# us the true optimal answer to compare the quantum result against.

best_assignment, best_cost = None, np.inf
for assignment in itertools.permutations(range(n_zones), n_uavs):
    cost = sum(distance[u, assignment[u]] for u in range(n_uavs))
    if cost < best_cost:
        best_cost, best_assignment = cost, assignment

print("\n--- Classical brute-force optimum ---")
for u in range(n_uavs):
    print(f"  {uav_names[u]} -> {zone_names[best_assignment[u]]}")
print(f"  Total distance: {best_cost:.1f} km")

# ---------------------------------------------------------------------------
# 3. QUANTUM APPROACH -- FORMULATE AS QUBO, SOLVE WITH QAOA
# ---------------------------------------------------------------------------
# One qubit per (UAV, zone) pair = "is UAV u assigned to zone z?" (1 or 0).
# With 2 UAVs x 3 zones that's 6 qubits.
#
# The cost function to minimize has two parts:
#   (a) total distance for whichever (UAV, zone) pairs are turned "on"
#   (b) a penalty that discourages assigning one UAV to zero or multiple
#       zones at once (each UAV should cover exactly one zone)
# This "cost + penalty" structure is exactly what QUBO problems look like
# in practice -- real constraints get folded into the objective as penalty
# terms since quantum optimizers work best on unconstrained problems.

n_qubits = n_uavs * n_zones  # one qubit per (UAV, zone) pair

# QAOA trains rotation angles whose magnitude is (angle x Hamiltonian
# coefficient) -- if the raw km distances feed in directly, coefficients in
# the hundreds make those rotations wrap around many times and the training
# landscape becomes needlessly jagged. Normalizing distances to O(1) before
# building the quantum Hamiltonian is standard practice and makes the
# circuit much easier to train; the *optimal assignment* is unaffected by
# this uniform rescaling, only the raw cost numbers are, so we keep the
# real km distances for reporting/plotting and only use the normalized
# version inside the quantum circuit.
distance_norm = distance / distance.max()
PENALTY = 1.5  # in normalized units; outweighs any single-swap saving (<=1)

def idx(u, z):
    return u * n_zones + z

def qubo_cost(bitstring, dist=distance_norm):
    """QUBO cost (normalized units) for a given bit assignment -- used to
    score/interpret QAOA's output, and to sanity-check the Hamiltonian
    below."""
    bits = np.array(bitstring).reshape(n_uavs, n_zones)
    cost = np.sum(bits * dist)
    # penalty: each UAV should be assigned to exactly one zone
    for u in range(n_uavs):
        cost += PENALTY * (bits[u].sum() - 1) ** 2
    return cost

dev = qml.device("default.qubit", wires=n_qubits)

# ---------------------------------------------------------------------------
# Build the QAOA cost Hamiltonian from the QUBO coefficients.
#
# Standard QUBO form: C(b) = sum_i diag_i * b_i + sum_{i<j} pair_ij * b_i*b_j
# (linear terms live on the diagonal, since b_i^2 = b_i for binary values).
# We first expand our cost (distance + one-hot penalty) into that form, then
# convert to an Ising Hamiltonian via b_i = (1 - Z_i) / 2, which is the
# standard substitution QAOA implementations use.
# ---------------------------------------------------------------------------
def build_qubo_matrix():
    diag = np.zeros(n_qubits)
    pair = np.zeros((n_qubits, n_qubits))  # only upper triangle (i<j) used
    const = 0.0  # constant offset (doesn't affect the optimal bitstring,
                 # kept only so printed costs match qubo_cost exactly)

    # Distance term: linear, one per (UAV, zone) qubit (normalized, see above)
    for u in range(n_uavs):
        for z in range(n_zones):
            diag[idx(u, z)] += distance_norm[u, z]

    # One-hot penalty per UAV: PENALTY * (sum_z b_z - 1)^2
    #   = -PENALTY * sum_z b_z + PENALTY * sum_{z<z'} 2*b_z*b_z' + PENALTY
    # (using b_z^2 = b_z to fold the squared diagonal term into the linear part)
    for u in range(n_uavs):
        ids = [idx(u, z) for z in range(n_zones)]
        for i in ids:
            diag[i] += -PENALTY
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                pair[i, j] += 2 * PENALTY
        const += PENALTY

    return diag, pair, const

diag, pair, const = build_qubo_matrix()

def qubo_from_matrix(bits_flat):
    """Cost from the matrix form -- used only to verify it matches qubo_cost."""
    c = float(np.dot(diag, bits_flat)) + const
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            c += pair[i, j] * bits_flat[i] * bits_flat[j]
    return c

# Sanity check: matrix form must reproduce the direct cost for every
# possible bitstring before we trust it inside the quantum circuit.
for i in range(2 ** n_qubits):
    bits = [(i >> b) & 1 for b in reversed(range(n_qubits))]
    a = qubo_cost(bits)
    c = qubo_from_matrix(np.array(bits))
    assert abs(a - c) < 1e-6, f"QUBO matrix mismatch at {bits}: {a} vs {c}"
print("QUBO matrix verified against direct cost function for all bitstrings.")

def build_cost_hamiltonian(diag, pair):
    coeffs, ops = [], []
    # Linear (Z_i) coefficients: h_i = -diag_i/2 - (1/4) * sum_j pair_sym[i,j]
    for i in range(n_qubits):
        neighbor_sum = sum(pair[i, j] for j in range(n_qubits) if j > i) + \
                        sum(pair[j, i] for j in range(n_qubits) if j < i)
        h_i = -diag[i] / 2 - neighbor_sum / 4
        coeffs.append(h_i)
        ops.append(qml.PauliZ(i))
    # Quadratic (Z_i Z_j) coefficients
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if pair[i, j] != 0:
                coeffs.append(pair[i, j] / 4)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, ops)

cost_h = build_cost_hamiltonian(diag, pair)
mixer_h = qaoa.x_mixer(range(n_qubits))

n_layers = 3  # QAOA "depth" -- more layers = more expressive, slower to train

def qaoa_layer(gamma, beta):
    qaoa.cost_layer(gamma, cost_h)
    qaoa.mixer_layer(beta, mixer_h)

@qml.qnode(dev)
def qaoa_circuit(params):
    for w in range(n_qubits):
        qml.Hadamard(wires=w)  # start in equal superposition of all assignments
    qml.layer(qaoa_layer, n_layers, params[0], params[1])
    return qml.probs(wires=range(n_qubits))

def expected_cost(params):
    probs = qaoa_circuit(params)
    total = 0.0
    for i, p in enumerate(probs):
        bits = [(i >> b) & 1 for b in reversed(range(n_qubits))]
        total += p * qubo_cost(bits)
    return total

# ---------------------------------------------------------------------------
# 4. TRAIN THE QAOA PARAMETERS
# ---------------------------------------------------------------------------
# QAOA on a hand-rolled QUBO landscape like this is sensitive to starting
# parameters (a known practical issue, not specific to this problem). We
# run a few random restarts with Adam and keep the best one -- a standard,
# honest way to handle that in a small demo.
N_RESTARTS = 5
N_STEPS = 150

best_params, best_final_cost = None, np.inf
print("\nTraining QAOA (multiple restarts)...")
for restart in range(N_RESTARTS):
    params = pnp.random.uniform(0, np.pi, (2, n_layers), requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=0.15)
    for step in range(N_STEPS):
        params = optimizer.step(expected_cost, params)
    final_cost = expected_cost(params)
    print(f"  restart {restart+1}: final expected cost {final_cost:.2f}")
    if final_cost < best_final_cost:
        best_final_cost, best_params = final_cost, params

params = best_params
print(f"Best restart expected cost: {best_final_cost:.2f}")

# ---------------------------------------------------------------------------
# 5. READ OUT THE MOST LIKELY ASSIGNMENT FROM THE TRAINED CIRCUIT
# ---------------------------------------------------------------------------
final_probs = qaoa_circuit(params)
best_i = np.argmax(final_probs)
best_bits = [(best_i >> b) & 1 for b in reversed(range(n_qubits))]
best_bits = np.array(best_bits).reshape(n_uavs, n_zones)

print("\n--- QAOA result (most likely bitstring) ---")
qaoa_cost_val = 0.0
qaoa_valid = True
for u in range(n_uavs):
    assigned = np.where(best_bits[u] == 1)[0]
    if len(assigned) == 1:
        z = assigned[0]
        print(f"  {uav_names[u]} -> {zone_names[z]}")
        qaoa_cost_val += distance[u, z]  # real km, for reporting
    else:
        qaoa_valid = False
        print(f"  {uav_names[u]} -> INVALID (penalty violated: {best_bits[u]})")
if qaoa_valid:
    print(f"  Total distance: {qaoa_cost_val:.1f} km")
else:
    print("  QAOA did not converge to a fully valid assignment on this run.")

# ---------------------------------------------------------------------------
# 6. PLOT -- MAP-STYLE COMPARISON
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))

for name, (lat, lon) in flood_zones.items():
    ax.scatter(lon, lat, c="crimson", s=140, marker="o", zorder=3)
    ax.annotate(name, (lon, lat), textcoords="offset points", xytext=(8, 5))

for name, (lat, lon) in uav_bases.items():
    ax.scatter(lon, lat, c="steelblue", s=200, marker="^", zorder=3)
    ax.annotate(name, (lon, lat), textcoords="offset points", xytext=(8, -12))

for u in range(n_uavs):
    z = best_assignment[u]
    lat1, lon1 = uav_bases[uav_names[u]]
    lat2, lon2 = flood_zones[zone_names[z]]
    ax.plot([lon1, lon2], [lat1, lat2], "b--", alpha=0.6, label="Classical optimum" if u == 0 else None)

ax.set_title("UAV -> Flood Zone Assignment (Classical Brute-Force Optimum)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
plt.tight_layout()
plt.savefig("uav_assignment.png", dpi=150)
print("\nSaved plot to uav_assignment.png")
