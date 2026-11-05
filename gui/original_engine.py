"""Raphael Langer's three functions, copied verbatim from Part 10.
Source commit: 430c3b01958f0a67dc2cd8352271f1c4b236378b.
The adapter supplies per-run globals; historical hardcoded summary fields in
get_expected_cost are ignored and recomputed for the selected scenario.
"""
import numpy as np
from itertools import combinations
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def get_QAOAcoefficients(lamda, p, d, constraint):
    K = lamda * constraint ** 2
    h = []
    Q = np.zeros((len(p), len(p)))
    for i,j in np.ndindex(Q.shape):
        if i < j:
            Q[i][j] = 2 * lamda * d[i] * d[j]
        elif i == j:
            Q[i][j] = p[i] + lamda * (d[i] ** 2 - 2 * constraint * d[i])
            h.append(Q[i][j])
        else:
            Q[i][j] = 0
    h = Q[np.diag_indices_from(Q)]
    J = np.triu(Q, k=1) # return only upper triangular portion o Q, everything else is 0
    
    # convert QUBO matrices into Ising coefficients
    alpha = [-h[i]/2 - (1/4) * sum([J[min(i,j)][max(i,j)] for j in range(len(h)) if j != i]) \
             for i in range(len(h))]
    beta  = J / 4
    c = K + (1/2) * sum(h) + (1/4) * J.sum()
    return alpha, beta, c, K, h, J

def get_QUBOcost(K, h, J, z):
    if len(h) == len(z) == len(J):
        pass
    else:
        raise ValueError("Error: Invalid QUBO inputs, dimension mismatch")
    z_pairs = list(combinations(range(len(z)), 2))    
    F = K + sum([h[i] * z[i] for i in range(len(h))]) + sum([J[i][j] * z[i] * z[j] for i,j in z_pairs])
    return F

def get_expected_cost(angles, num_qubits, alpha, beta): # angles is a list: [gamma1, beta_mixer1, gamma2, beta_mixer2...]
    angles = np.array(angles).reshape(-1,2) # format angles into pairs, one pair per layer
    p = len(angles) # p = number of layers
    
    # build quantum circuit
    qc = QuantumCircuit(num_qubits) # 10-qubit circuit
    qc.h(range(num_qubits)) # prepare each qubit in the |+> state = equal superposition
    qc.barrier()
    for layer in range(p):
        gamma, beta_mixer = angles[layer]

        # cost unitary
        for i in range(num_qubits):
            qc.rz(2 * gamma * alpha[i], i) # first step of cost unitary - single-qubit coefficients
        for i,j in pairs:
            qc.rzz(2 * gamma * beta[i][j], i, j) # second step of cost unitary - pairwise coefficients
        qc.barrier()
        
        # mixer unitary
        for i in range(num_qubits):
            qc.rx(2 * beta_mixer, i)
    
    # run circuit on |0000000000> to get resulting amplitudes for each possible assignment
    state_result = Statevector.from_instruction(qc)
    probabilities = state_result.probabilities() # list of squared magnitudes

    # for debugging:
    #print()
    #print("probability of optimal HVAC state:", probabilities[4])
    #print("most probable state = ", total_assignments[np.argmax(probabilities)])
    #print("probability of most probable state:", np.max(probabilities))
    #print()
    
    expected_cost = np.dot(probabilities, QUBOcosts)
    p_optimal = f"{round(100 * probabilities[4], 5)}%" # known optimal is |0000000100> = C ON, everything else OFF
    most_likely = (total_assignments[np.argmax(probabilities)], round(100 * float(max(probabilities)), 5))
    prob_feasible = 0
    for i in range(len(total_assignments)):
        if total_assignments[i][0] * 50 + total_assignments[i][1] * 75 + total_assignments[i][2] * 100 >= 100:
            prob_feasible += probabilities[i]
    prob_feasible = f"{round(100 * prob_feasible, 5)}%"
    
    return expected_cost, state_result, probabilities, p_optimal, most_likely, prob_feasible, qc
