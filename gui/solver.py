"""GUI adapter around Raphael's original QUBO and Qiskit functions."""
import math
import time
from itertools import product, combinations
from types import FunctionType
import numpy as np
from scipy.optimize import minimize
try:
    from . import original_engine
except ImportError:
    import original_engine


def number(data, key, default, low, high, integer=False):
    value = float(data.get(key, default))
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{key} must be between {low} and {high}.')
    if integer and not value.is_integer():
        raise ValueError(f'{key} must be an integer.')
    return int(value) if integer else value


def solve(data):
    start = time.perf_counter()
    capacities = [number(data, f'capacity{i}', v, 1, 150, True) for i, v in enumerate([50, 75, 100])]
    powers = [number(data, f'power{i}', v, 0.1, 100) for i, v in enumerate([10, 17, 25])]
    demand = number(data, 'demand', 100, 1, 450, True)
    if demand > sum(capacities):
        raise ValueError('Cooling demand exceeds the combined equipment capacity. Lower demand or increase capacity.')
    penalty = number(data, 'penalty', 1, 0.01, 100)
    scale = number(data, 'scale', 10000, 1, 1000000)
    layers = number(data, 'layers', 1, 1, 5, True)
    iterations = number(data, 'iterations', 120, 20, 500, True)
    shots = number(data, 'shots', 2000, 100, 10000, True)
    seed = number(data, 'seed', 42, 0, 2147483647, True)
    gamma = number(data, 'gamma', 0.01, -100, 100)
    mixer = number(data, 'mixer', 0.02, -100, 100)
    # Exact integer slack covers every possible cooling surplus, including zero.
    surplus = sum(capacities) - demand
    slack_bits = surplus.bit_length()
    diff = capacities + [-2**i for i in range(slack_bits)]
    power = powers + [0] * slack_bits
    n = len(diff)
    assignments = np.array(list(product([0, 1], repeat=n)), dtype=int)[:, ::-1]
    pairs = list(combinations(range(n), 2))
    alpha, beta, c, K, h, J = original_engine.get_QAOAcoefficients(penalty, power, diff, demand)
    costs = np.array([original_engine.get_QUBOcost(K, h, J, z) for z in assignments])
    # Each run gets its own namespace; the original functions remain unchanged.
    context = dict(original_engine.__dict__)
    context.update(pairs=pairs, total_assignments=assignments.tolist(), QUBOcosts=costs)
    evaluate = FunctionType(original_engine.get_expected_cost.__code__, context)
    cooling = assignments[:, :3] @ capacities
    consumption = assignments[:, :3] @ powers
    feasible = cooling >= demand
    configs = np.array(list(product([0, 1], repeat=3)))
    config_cooling = configs @ capacities
    config_power = configs @ powers
    best_power = float(min(config_power[config_cooling >= demand]))
    optimal = feasible & np.isclose(consumption, best_power, rtol=0, atol=1e-9)
    slack = assignments[:, 3:] @ np.array([2**i for i in range(slack_bits)], dtype=int)
    exact_constraint = cooling - slack == demand
    trace = []

    def objective(angles):
        cost = float(evaluate(angles, n, np.array(alpha) / scale, beta / scale)[0])
        trace.append(cost)
        return cost

    initial = [gamma, mixer] + [0.1, 0.1] * (layers - 1)
    qaoa_start = time.perf_counter()
    result = minimize(objective, initial, method='COBYLA', options={'maxiter': iterations})
    final = evaluate(result.x, n, np.array(alpha) / scale, beta / scale)
    probs = final[2]
    qc = final[6]
    qaoa_seconds = time.perf_counter() - qaoa_start
    counts = np.random.default_rng(seed).multinomial(shots, probs / probs.sum())
    equipment_prob = np.zeros(8)
    equipment_counts = np.zeros(8, dtype=int)
    codes = assignments[:, :3] @ [4, 2, 1]
    for i in range(8):
        equipment_prob[i] = probs[codes == i].sum()
        equipment_counts[i] = counts[codes == i].sum()
    sampled_feasible = np.flatnonzero((equipment_counts > 0) & (config_cooling >= demand))
    best_sample = int(min(sampled_feasible, key=lambda i: config_power[i])) if len(sampled_feasible) else None
    baseline_choices = np.flatnonzero((config_cooling >= demand) & np.isclose(config_power, best_power, rtol=0, atol=1e-9))
    rows = []
    for i, config in enumerate(configs):
        rows.append(dict(bits=''.join(map(str, config)), units=[label for label, bit in zip('ABC', config) if bit],
                         cooling=int(config_cooling[i]), power=float(config_power[i]), feasible=bool(config_cooling[i] >= demand),
                         optimal=bool(i in baseline_choices), probability=float(equipment_prob[i]), count=int(equipment_counts[i])))
    full_states = []
    for index in np.argsort(probs)[-10:][::-1]:
        full_states.append(dict(bitstring=format(int(index), f'0{n}b'), probability=float(probs[index]),
                                cost=float(costs[index]), slack=int(slack[index])))
    return dict(rows=rows, baseline=rows[int(baseline_choices[0])], best_sample=None if best_sample is None else rows[best_sample],
                feasible_probability=float(probs[feasible].sum()), optimal_equipment_probability=float(probs[optimal].sum()),
                optimal_full_probability=float(probs[optimal & exact_constraint].sum()),
                constraint_probability=float(probs[exact_constraint].sum()), sample_feasible=float(counts[feasible].sum() / shots),
                expected_cost=float(final[0]), uniform_cost=float(costs.mean()), sampled_cost=float(counts @ costs / shots),
                expected_power=float(probs @ consumption), baseline_power=best_power, trace=trace,
                angles=result.x.reshape(-1, 2).tolist(), evaluations=int(result.nfev), converged=bool(result.success),
                message=str(result.message), qubits=n, slack_bits=slack_bits, scale=scale,
                circuit_depth=int(qc.depth()), two_qubit_gates=sum(inst.operation.num_qubits == 2 for inst in qc.data),
                qaoa_seconds=qaoa_seconds, total_seconds=time.perf_counter() - start,
                shots=shots, seed=seed, full_states=full_states, circuit=str(qc.draw(output='text', fold=110)))
