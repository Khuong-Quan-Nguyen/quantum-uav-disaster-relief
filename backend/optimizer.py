"""
Core optimization logic for the quantum UAV disaster-relief demo.
Extracted from the original standalone script so it can be called from the
FastAPI backend. Same math, same verification step -- just wrapped as
functions instead of top-level script code, and sped up (lightning.qubit
device, fewer restarts/layers) so it can run in a few seconds behind an API
instead of ~40s as a one-off script.
"""

import itertools
import time
import numpy as np
import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as pnp

FLOOD_ZONES = {
    "Quang Binh":     (17.47, 106.62),
    "Quang Tri":      (16.75, 107.19),
    "Thua Thien Hue": (16.47, 107.59),
}

UAV_BASES = {
    "UAV-1": (17.85, 106.35),
    "UAV-2": (16.30, 107.70),
}


def haversine_km(p1, p2):
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def build_distance_matrix(zones, bases):
    zone_names = list(zones.keys())
    uav_names = list(bases.keys())
    distance = np.array([
        [haversine_km(bases[u], zones[z]) for z in zone_names]
        for u in uav_names
    ])
    return distance, uav_names, zone_names


def classical_optimum(distance, n_uavs, n_zones):
    best_assignment, best_cost = None, np.inf
    for assignment in itertools.permutations(range(n_zones), n_uavs):
        cost = sum(distance[u, assignment[u]] for u in range(n_uavs))
        if cost < best_cost:
            best_cost, best_assignment = cost, assignment
    return best_assignment, best_cost


def run_quantum_optimization(progress_cb=None):
    """
    Runs the full pipeline: build scenario -> classical brute force ->
    QUBO construction (verified) -> QAOA training (multi-restart) ->
    readout. Calls progress_cb(fraction, message) periodically if given,
    so a caller (e.g. a background job) can report live progress.
    """
    def report(frac, msg):
        if progress_cb:
            progress_cb(frac, msg)

    report(0.02, "Building disaster-relief scenario")
    distance, uav_names, zone_names = build_distance_matrix(FLOOD_ZONES, UAV_BASES)
    n_uavs, n_zones = len(uav_names), len(zone_names)

    report(0.08, "Solving classical brute-force baseline")
    best_assignment, best_cost = classical_optimum(distance, n_uavs, n_zones)

    n_qubits = n_uavs * n_zones

    def idx(u, z):
        return u * n_zones + z

    distance_norm = distance / distance.max()
    PENALTY = 1.5

    def qubo_cost(bitstring, dist=distance_norm):
        bits = np.array(bitstring).reshape(n_uavs, n_zones)
        cost = np.sum(bits * dist)
        for u in range(n_uavs):
            cost += PENALTY * (bits[u].sum() - 1) ** 2
        return cost

    report(0.15, "Formulating QUBO (one qubit per UAV-zone pair)")

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
                i, j = ids[a], ids[b]
                pair[i, j] += 2 * PENALTY
        const += PENALTY

    def qubo_from_matrix(bits_flat):
        c = float(np.dot(diag, bits_flat)) + const
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                c += pair[i, j] * bits_flat[i] * bits_flat[j]
        return c

    report(0.22, "Verifying QUBO against brute-force cost for all bitstrings")
    for i in range(2 ** n_qubits):
        bits = [(i >> b) & 1 for b in reversed(range(n_qubits))]
        assert abs(qubo_cost(bits) - qubo_from_matrix(np.array(bits))) < 1e-6

    coeffs, ops = [], []
    for i in range(n_qubits):
        neighbor_sum = sum(pair[i, j] for j in range(n_qubits) if j > i) + \
                        sum(pair[j, i] for j in range(n_qubits) if j < i)
        h_i = -diag[i] / 2 - neighbor_sum / 4
        coeffs.append(h_i)
        ops.append(qml.PauliZ(i))
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if pair[i, j] != 0:
                coeffs.append(pair[i, j] / 4)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    cost_h = qml.Hamiltonian(coeffs, ops)
    mixer_h = qaoa.x_mixer(range(n_qubits))

    n_layers = 2
    dev = qml.device("lightning.qubit", wires=n_qubits)

    def qaoa_layer(gamma, beta):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(beta, mixer_h)

    def circuit_ansatz(params):
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])

    # Training uses expval(cost_h) directly (fast, adjoint-differentiable
    # on the lightning simulator). Readout uses a separate probs-based
    # qnode (forward-only, no differentiation needed) to recover the most
    # likely bitstring.
    @qml.qnode(dev, diff_method="adjoint")
    def qaoa_expval(params):
        circuit_ansatz(params)
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def qaoa_probs(params):
        circuit_ansatz(params)
        return qml.probs(wires=range(n_qubits))

    def expected_cost(params):
        # qml.expval(cost_h) already IS the expected QUBO cost (up to the
        # constant offset dropped during Hamiltonian construction), so no
        # manual probability-weighted sum is needed here.
        return qaoa_expval(params) + const

    N_RESTARTS = 3
    N_STEPS = 60
    best_params, best_final_cost = None, np.inf
    restart_log = []

    report(0.3, "Training QAOA parameters")
    for restart in range(N_RESTARTS):
        params = pnp.random.uniform(0, np.pi, (2, n_layers), requires_grad=True)
        optimizer = qml.AdamOptimizer(stepsize=0.15)
        for step in range(N_STEPS):
            params = optimizer.step(expected_cost, params)
            frac = 0.3 + 0.6 * ((restart * N_STEPS + step + 1) / (N_RESTARTS * N_STEPS))
            report(frac, f"QAOA restart {restart+1}/{N_RESTARTS}, step {step+1}/{N_STEPS}")
        final_cost = expected_cost(params)
        restart_log.append({"restart": restart + 1, "final_expected_cost": float(final_cost)})
        if final_cost < best_final_cost:
            best_final_cost, best_params = final_cost, params

    report(0.95, "Reading out best assignment")
    final_probs = qaoa_probs(best_params)
    best_i = int(np.argmax(final_probs))
    best_bits = [(best_i >> b) & 1 for b in reversed(range(n_qubits))]
    best_bits = np.array(best_bits).reshape(n_uavs, n_zones)

    qaoa_assignment = {}
    qaoa_valid = True
    qaoa_cost_val = 0.0
    for u in range(n_uavs):
        assigned = np.where(best_bits[u] == 1)[0]
        if len(assigned) == 1:
            z = int(assigned[0])
            qaoa_assignment[uav_names[u]] = zone_names[z]
            qaoa_cost_val += float(distance[u, z])
        else:
            qaoa_valid = False
            qaoa_assignment[uav_names[u]] = None

    classical_assignment = {
        uav_names[u]: zone_names[best_assignment[u]] for u in range(n_uavs)
    }

    report(1.0, "Done")

    return {
        "flood_zones": {name: {"lat": lat, "lon": lon} for name, (lat, lon) in FLOOD_ZONES.items()},
        "uav_bases": {name: {"lat": lat, "lon": lon} for name, (lat, lon) in UAV_BASES.items()},
        "distance_matrix": distance.tolist(),
        "uav_names": uav_names,
        "zone_names": zone_names,
        "classical": {
            "assignment": classical_assignment,
            "total_km": float(best_cost),
        },
        "quantum": {
            "assignment": qaoa_assignment,
            "total_km": qaoa_cost_val if qaoa_valid else None,
            "valid": qaoa_valid,
            "restart_log": restart_log,
        },
        "qubits_used": n_qubits,
    }


if __name__ == "__main__":
    t0 = time.time()
    result = run_quantum_optimization(progress_cb=lambda f, m: print(f"{f*100:5.1f}%  {m}"))
    print(f"\nDone in {time.time()-t0:.1f}s")
    print(result["classical"])
    print(result["quantum"])
