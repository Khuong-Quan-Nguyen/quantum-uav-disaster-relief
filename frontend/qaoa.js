/**
 * Pure-JavaScript QAOA statevector simulator for the UAV-to-zone assignment QUBO.
 *
 * This is a from-scratch port of backend/optimizer.py. Six qubits is 64
 * amplitudes, so the whole optimization runs client-side in a few tens of
 * milliseconds -- no Python, no backend, no PennyLane. The PennyLane version
 * is kept as the reference implementation; tools/crosscheck.py verifies this
 * file against it (see README).
 *
 * Conventions match PennyLane so the two can be compared directly:
 *   - wire 0 is the MOST significant bit of a basis-state index
 *   - cost layer  = exp(-i*gamma*H_C), exact since every term is diagonal in Z
 *   - mixer layer = exp(-i*beta*sum_i X_i), i.e. RX(2*beta) on each wire
 */
const QAOA = (function () {

  const FLOOD_ZONES = {
    "Quang Binh":     [17.47, 106.62],
    "Quang Tri":      [16.75, 107.19],
    "Thua Thien Hue": [16.47, 107.59],
  };

  const UAV_BASES = {
    "UAV-1": [17.85, 106.35],
    "UAV-2": [16.30, 107.70],
  };

  const PENALTY = 1.5;
  const N_LAYERS = 2;
  const N_RESTARTS = 3;
  const N_STEPS = 60;
  const STEPSIZE = 0.15;

  // ---------------------------------------------------------------- geometry

  function haversineKm(p1, p2) {
    const r = Math.PI / 180;
    const lat1 = p1[0] * r, lon1 = p1[1] * r;
    const lat2 = p2[0] * r, lon2 = p2[1] * r;
    const dlat = lat2 - lat1, dlon = lon2 - lon1;
    const a = Math.sin(dlat / 2) ** 2 +
              Math.cos(lat1) * Math.cos(lat2) * Math.sin(dlon / 2) ** 2;
    return 2 * 6371 * Math.asin(Math.sqrt(a));
  }

  function buildDistanceMatrix(zones, bases) {
    const zoneNames = Object.keys(zones);
    const uavNames = Object.keys(bases);
    const distance = uavNames.map(u => zoneNames.map(z => haversineKm(bases[u], zones[z])));
    return { distance, uavNames, zoneNames };
  }

  /** Brute-force optimum: every injective UAV -> zone assignment. */
  function classicalOptimum(distance, nUavs, nZones) {
    let bestAssignment = null, bestCost = Infinity;
    const chosen = [], used = new Array(nZones).fill(false);
    (function recurse(u, cost) {
      if (u === nUavs) {
        if (cost < bestCost) { bestCost = cost; bestAssignment = chosen.slice(); }
        return;
      }
      for (let z = 0; z < nZones; z++) {
        if (used[z]) continue;
        used[z] = true; chosen.push(z);
        recurse(u + 1, cost + distance[u][z]);
        chosen.pop(); used[z] = false;
      }
    })(0, 0);
    return { assignment: bestAssignment, cost: bestCost };
  }

  // ------------------------------------------------------------------- QUBO

  /**
   * One binary variable per (UAV, zone) pair. Objective is total distance,
   * plus PENALTY * (sum_z x[u][z] - 1)^2 per UAV to enforce "exactly one zone".
   */
  function buildQubo(distance, nUavs, nZones) {
    const nQubits = nUavs * nZones;
    const idx = (u, z) => u * nZones + z;

    let distMax = 0;
    for (const row of distance) for (const d of row) if (d > distMax) distMax = d;
    const distanceNorm = distance.map(row => row.map(d => d / distMax));

    const diag = new Float64Array(nQubits);
    const pair = new Float64Array(nQubits * nQubits); // upper triangle only
    let constTerm = 0;

    for (let u = 0; u < nUavs; u++)
      for (let z = 0; z < nZones; z++)
        diag[idx(u, z)] += distanceNorm[u][z];

    for (let u = 0; u < nUavs; u++) {
      const ids = [];
      for (let z = 0; z < nZones; z++) ids.push(idx(u, z));
      for (const i of ids) diag[i] += -PENALTY;
      for (let a = 0; a < ids.length; a++)
        for (let b = a + 1; b < ids.length; b++)
          pair[ids[a] * nQubits + ids[b]] += 2 * PENALTY;
      constTerm += PENALTY;
    }

    return { nQubits, idx, distanceNorm, diag, pair, constTerm };
  }

  /** Direct evaluation of the QUBO from a bit array -- the ground truth. */
  function quboCostDirect(bits, distanceNorm, nUavs, nZones) {
    let cost = 0;
    for (let u = 0; u < nUavs; u++) {
      let rowSum = 0;
      for (let z = 0; z < nZones; z++) {
        cost += bits[u * nZones + z] * distanceNorm[u][z];
        rowSum += bits[u * nZones + z];
      }
      cost += PENALTY * (rowSum - 1) ** 2;
    }
    return cost;
  }

  // ------------------------------------------------------------------ Ising

  /**
   * QUBO -> Ising via x_i = (1 - z_i)/2.
   *
   * Note the full constant: the Python version folds only `constTerm` back in,
   * which leaves its reported expected costs shifted by `offset`. Adding both
   * here makes the reported energy an actual expected QUBO cost. The shift
   * never affected the argmin, so the assignments agree either way.
   */
  function buildIsing(qubo) {
    const { nQubits, diag, pair, constTerm } = qubo;
    const h = new Float64Array(nQubits);
    const J = new Float64Array(nQubits * nQubits);

    let sumDiag = 0, sumPair = 0;
    for (let i = 0; i < nQubits; i++) sumDiag += diag[i];

    for (let i = 0; i < nQubits; i++) {
      let neighborSum = 0;
      for (let j = i + 1; j < nQubits; j++) neighborSum += pair[i * nQubits + j];
      for (let j = 0; j < i; j++) neighborSum += pair[j * nQubits + i];
      h[i] = -diag[i] / 2 - neighborSum / 4;
    }
    for (let i = 0; i < nQubits; i++)
      for (let j = i + 1; j < nQubits; j++) {
        const p = pair[i * nQubits + j];
        if (p !== 0) { J[i * nQubits + j] = p / 4; sumPair += p; }
      }

    const offset = sumDiag / 2 + sumPair / 4;
    return { h, J, offset, energyConst: constTerm + offset };
  }

  /** bit i of basis index `state`, with wire 0 as the most significant bit. */
  function bitOf(state, i, nQubits) {
    return (state >> (nQubits - 1 - i)) & 1;
  }

  /** Diagonal of H_C over every basis state -- precomputed once, reused every step. */
  function buildEnergies(ising, nQubits) {
    const { h, J } = ising;
    const dim = 1 << nQubits;
    const energies = new Float64Array(dim);
    const z = new Float64Array(nQubits);
    for (let s = 0; s < dim; s++) {
      for (let i = 0; i < nQubits; i++) z[i] = 1 - 2 * bitOf(s, i, nQubits);
      let e = 0;
      for (let i = 0; i < nQubits; i++) {
        e += h[i] * z[i];
        for (let j = i + 1; j < nQubits; j++) {
          const Jij = J[i * nQubits + j];
          if (Jij !== 0) e += Jij * z[i] * z[j];
        }
      }
      energies[s] = e;
    }
    return energies;
  }

  // -------------------------------------------------------------- simulator

  function simulate(gammas, betas, energies, nQubits) {
    const dim = 1 << nQubits;
    const re = new Float64Array(dim);
    const im = new Float64Array(dim);
    re.fill(1 / Math.sqrt(dim)); // Hadamard on every wire

    for (let l = 0; l < gammas.length; l++) {
      // cost layer: exp(-i*gamma*E) is a per-basis-state phase
      const g = gammas[l];
      for (let s = 0; s < dim; s++) {
        const phase = -g * energies[s];
        const c = Math.cos(phase), sn = Math.sin(phase);
        const r = re[s], i2 = im[s];
        re[s] = r * c - i2 * sn;
        im[s] = r * sn + i2 * c;
      }
      // mixer layer: RX(2*beta) on each wire
      const b = betas[l];
      const c = Math.cos(b), sn = Math.sin(b);
      for (let w = 0; w < nQubits; w++) {
        const stride = 1 << (nQubits - 1 - w);
        for (let base = 0; base < dim; base += stride << 1) {
          for (let k = base; k < base + stride; k++) {
            const k2 = k + stride;
            const aRe = re[k],  aIm = im[k];
            const bRe = re[k2], bIm = im[k2];
            re[k]  =  c * aRe + sn * bIm;
            im[k]  =  c * aIm - sn * bRe;
            re[k2] =  sn * aIm + c * bRe;
            im[k2] = -sn * aRe + c * bIm;
          }
        }
      }
    }
    return { re, im };
  }

  function probsOf(state) {
    const { re, im } = state;
    const p = new Float64Array(re.length);
    for (let s = 0; s < re.length; s++) p[s] = re[s] * re[s] + im[s] * im[s];
    return p;
  }

  /** <H_C> for the given parameters (no constant folded in). */
  function expval(params, energies, nQubits) {
    const p = probsOf(simulate(params.gammas, params.betas, energies, nQubits));
    let e = 0;
    for (let s = 0; s < p.length; s++) e += p[s] * energies[s];
    return e;
  }

  // -------------------------------------------------------------- optimizer

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Central-difference gradient. Four parameters, so this is cheap and stable. */
  function gradient(flat, objective) {
    const H = 1e-6;
    const g = new Float64Array(flat.length);
    for (let i = 0; i < flat.length; i++) {
      const orig = flat[i];
      flat[i] = orig + H; const up = objective(flat);
      flat[i] = orig - H; const dn = objective(flat);
      flat[i] = orig;
      g[i] = (up - dn) / (2 * H);
    }
    return g;
  }

  /** Adam, matching PennyLane's AdamOptimizer defaults (beta1=0.9, beta2=0.99). */
  function adamStep(flat, grad, state, stepsize) {
    const b1 = 0.9, b2 = 0.99, eps = 1e-8;
    state.t += 1;
    for (let i = 0; i < flat.length; i++) {
      state.m[i] = b1 * state.m[i] + (1 - b1) * grad[i];
      state.v[i] = b2 * state.v[i] + (1 - b2) * grad[i] * grad[i];
      const mHat = state.m[i] / (1 - Math.pow(b1, state.t));
      const vHat = state.v[i] / (1 - Math.pow(b2, state.t));
      flat[i] -= stepsize * mHat / (Math.sqrt(vHat) + eps);
    }
  }

  // ------------------------------------------------------------------- API

  /**
   * Full pipeline. Mirrors run_quantum_optimization() in backend/optimizer.py.
   * `onProgress(fraction, message)` is called with the optimizer's real state.
   * Async only so the browser can repaint the console between steps.
   */
  async function run(options) {
    const opts = options || {};
    const onProgress = opts.onProgress || function () {};
    const yieldEvery = opts.yieldEvery === undefined ? 6 : opts.yieldEvery;
    const rand = mulberry32(opts.seed === undefined ? (Date.now() & 0xffffffff) : opts.seed);
    const breathe = () => new Promise(r => setTimeout(r, 0));

    onProgress(0.02, "Building disaster-relief scenario");
    const { distance, uavNames, zoneNames } = buildDistanceMatrix(FLOOD_ZONES, UAV_BASES);
    const nUavs = uavNames.length, nZones = zoneNames.length;

    onProgress(0.08, "Solving classical brute-force baseline");
    const classical = classicalOptimum(distance, nUavs, nZones);

    onProgress(0.15, "Formulating QUBO (one qubit per UAV-zone pair)");
    const qubo = buildQubo(distance, nUavs, nZones);
    const nQubits = qubo.nQubits;
    const ising = buildIsing(qubo);
    const energies = buildEnergies(ising, nQubits);

    // The signature check: the Ising diagonal must reproduce the QUBO exactly
    // for every one of the 2^n bitstrings before any of it is trusted.
    onProgress(0.22, "Verifying Ising diagonal against QUBO for all 2^" + nQubits + " bitstrings");
    const dim = 1 << nQubits;
    let maxErr = 0;
    const bits = new Array(nQubits);
    for (let s = 0; s < dim; s++) {
      for (let i = 0; i < nQubits; i++) bits[i] = bitOf(s, i, nQubits);
      const direct = quboCostDirect(bits, qubo.distanceNorm, nUavs, nZones);
      const viaIsing = energies[s] + ising.energyConst;
      maxErr = Math.max(maxErr, Math.abs(direct - viaIsing));
    }
    if (maxErr > 1e-9) throw new Error("QUBO/Ising mismatch: " + maxErr);
    onProgress(0.25, "Verification passed (max error " + maxErr.toExponential(2) + ")");

    const objective = flat => expval(
      { gammas: flat.slice(0, N_LAYERS), betas: flat.slice(N_LAYERS) },
      energies, nQubits
    ) + ising.energyConst;

    onProgress(0.3, "Training QAOA parameters");
    let bestFlat = null, bestFinalCost = Infinity;
    const restartLog = [];

    for (let restart = 0; restart < N_RESTARTS; restart++) {
      const flat = new Float64Array(2 * N_LAYERS);
      for (let i = 0; i < flat.length; i++) flat[i] = rand() * Math.PI;
      const adam = { m: new Float64Array(flat.length), v: new Float64Array(flat.length), t: 0 };

      for (let step = 0; step < N_STEPS; step++) {
        adamStep(flat, gradient(flat, objective), adam, STEPSIZE);
        const frac = 0.3 + 0.6 * ((restart * N_STEPS + step + 1) / (N_RESTARTS * N_STEPS));
        // Log every few steps: 180 lines of "step k/60" buries the real signal.
        if ((step + 1) % 10 === 0 || step === N_STEPS - 1) {
          onProgress(frac, "QAOA restart " + (restart + 1) + "/" + N_RESTARTS +
                           ", step " + (step + 1) + "/" + N_STEPS +
                           "  <E> = " + objective(flat).toFixed(4));
        }
        if (yieldEvery && step % yieldEvery === 0) await breathe();
      }

      const finalCost = objective(flat);
      restartLog.push({ restart: restart + 1, final_expected_cost: finalCost });
      onProgress(0.9, "Restart " + (restart + 1) + " converged to <E> = " + finalCost.toFixed(4));
      if (finalCost < bestFinalCost) { bestFinalCost = finalCost; bestFlat = flat.slice(); }
    }

    onProgress(0.95, "Reading out best assignment");
    const finalProbs = probsOf(simulate(
      bestFlat.slice(0, N_LAYERS), bestFlat.slice(N_LAYERS), energies, nQubits));
    let bestState = 0;
    for (let s = 1; s < finalProbs.length; s++) if (finalProbs[s] > finalProbs[bestState]) bestState = s;

    const assignment = {};
    let valid = true, totalKm = 0;
    for (let u = 0; u < nUavs; u++) {
      const on = [];
      for (let z = 0; z < nZones; z++) if (bitOf(bestState, u * nZones + z, nQubits) === 1) on.push(z);
      if (on.length === 1) {
        assignment[uavNames[u]] = zoneNames[on[0]];
        totalKm += distance[u][on[0]];
      } else {
        valid = false;
        assignment[uavNames[u]] = null;
      }
    }

    const coveredZones = new Set(Object.values(assignment).filter(Boolean));
    const uncovered = zoneNames.filter(z => !coveredZones.has(z));

    onProgress(1.0, "Done");

    return {
      flood_zones: Object.fromEntries(Object.entries(FLOOD_ZONES).map(([n, p]) => [n, { lat: p[0], lon: p[1] }])),
      uav_bases:   Object.fromEntries(Object.entries(UAV_BASES).map(([n, p]) => [n, { lat: p[0], lon: p[1] }])),
      distance_matrix: distance,
      uav_names: uavNames,
      zone_names: zoneNames,
      classical: {
        assignment: Object.fromEntries(classical.assignment.map((z, u) => [uavNames[u], zoneNames[z]])),
        total_km: classical.cost,
      },
      quantum: {
        assignment,
        total_km: valid ? totalKm : null,
        valid,
        readout_probability: finalProbs[bestState],
        restart_log: restartLog,
      },
      uncovered_zones: uncovered,
      qubits_used: nQubits,
      verification_max_error: maxErr,
    };
  }

  return {
    FLOOD_ZONES, UAV_BASES, PENALTY, N_LAYERS,
    haversineKm, buildDistanceMatrix, classicalOptimum,
    buildQubo, quboCostDirect, buildIsing, buildEnergies, bitOf,
    simulate, probsOf, expval, run,
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = QAOA;
