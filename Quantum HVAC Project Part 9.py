

# In[ ]:


# module 9

print("PART A - EXPLAIN THE ALGORITHM\n")

print('''the cost hamiltonian gives us basis-state costs as eigenvalues. the cost unitary is the 
matrix exponentiation of the cost hamiltonian, so it is a unitary operation that can be applied
to a quantum state and preserve normalization. this operation essentially changes the 
phase of each basis state based on its cost. the mixer hamiltonian is a matrix that
combines the individual X (NOT) gates for each qubit into one large matrix. the mixer unitary 
is the matrix exponentiation of the mixer hamiltonian, so it is a unitary operation; it can be 
applied to a quantum state to mix amplitudes, so that different basis states will now have 
amplitudes that are dependent on each other. The variational angles are the tuneable parameters 
that govern the cost unitary and mixer unitary functions and influence how effective the QAOA is.
the classical optimizer is a non-Quantum optimizer loop that finds the optimal variational angles
for the QAOA objective. The cost layer alone only changes the phases of the basis states, so their
respective probabilities remain the same until the mixer is applied.''')


# In[22]:


print("PART B - SINGLE QUBIT GATE CHECKS\n")

import math
import numpy as np
import random
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

test_alpha = 5
test_beta = random.uniform(0, 2 * math.pi)
test_gamma = random.uniform(0, 2 * math.pi)

print(f"alpha = {test_alpha}\n\nbeta = {test_beta}\n\ngamma = {test_gamma}\n\n")

# Apply RZ and RX manually
Z = Operator([[1, 0], [0, -1]])
X = Operator([[0, 1], [1, 0]])

def RZ(gamma, alpha):
    return expm(-1j * gamma * alpha * Z.data)  # theta / 2 = gamma x alpha

def RX(beta):
    return expm(-1j * beta * X.data) # theta = 2 x beta 

test_qubit = Statevector([1/np.sqrt(2), 1/np.sqrt(2)]) # |+>
manual_RZ = RZ(test_gamma, test_alpha) @ test_qubit.data
manual_RX = RX(test_beta) @ test_qubit.data

# apply RZ and RX through qiskit built-in functions
qc1 = QuantumCircuit(1)
qc1.h(0) # puts qubit 0 into the |+> state
qc1.rz(2*test_gamma*test_alpha, 0)
qiskit_RZ = Statevector.from_instruction(qc1)

qc2 = QuantumCircuit(1)
qc2.h(0) # puts qubit 0 into the |+> state
qc2.rx(2*test_beta, 0)
qiskit_RX = Statevector.from_instruction(qc2)

print(f"Manual RZ = e^(-i x gamma x alpha x Z) = {manual_RZ}")
print(f"Qiskit RZ = {qiskit_RZ.data}")
print() 
print(f"Manual RX = e^(-i x beta x X) = {manual_RX}")
print(f"Qiskit RX = {qiskit_RX.data}")


# In[15]:


print("PART C - TWO QUBIT ZZ CHECK\n")

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp
from itertools import combinations, product
from scipy import linalg

print(''' f(u,v) = 3u + 2v + 4uv + 5
 u = (1 - s1) / 2; v = (1 - s2)/2
 f(s1, s2) = 3/2 - (3/2)s1 + 1 - s2 + 1 - s1 - s2 + s1s2 + 5
 f(s1,s2) = (-5/2)s1 + (-2)s2 + (1)s1s2 + (17/2)
 c = 17/2
 alpha1 = -5/2; alpha2 = -2
 beta12 = 1 ''')

print()
print()

# define bit
bit = [0, 1]
num_qubits = 2 # u and v
compound_states = list(product(bit, repeat=num_qubits))

# ising coefficients
c = 17 / 2
alpha = [-5/2, -2]
beta = np.array([[0, 1],
                [0, 0]])

# define variatonal angles
gamma = 0.2

# build quantum circuit

pairs = list(combinations(range(num_qubits), 2)) # all index pairs where i < j
qc = QuantumCircuit(num_qubits) 
for i in range(len(alpha)):
    qc.rz(2 * gamma * alpha[i], i)
for i,j in pairs:
    qc.rzz(2 * gamma * beta[i][j], i, j)
# qc.global_phase += -gamma * c # if this is added, no need to test for global phase difference below

# F(z) QUBO formulation
def test_ising(s1, s2): # s1 and s2 are Pauli variables --> -1 or +1
    # get combinations of indices - e.g. (0,1) = (s1,s2)
    s = [s1, s2]
    s_pairs = list(combinations(range(len(s)), 2))
    f = c + sum([alpha[i] * s[i] for i in range(len(s))]) + \
    sum([beta[i][j] * s[i] * s[j] for i,j in s_pairs])
    return f

# test the circuit and compare to F(z) QUBO formulation
bit = [0, 1]
for i in range(len(bit) ** num_qubits):
    # circuit method
    basis_state = [0] * len(bit) ** num_qubits
    basis_state[i] = 1
    compound_state = compound_states[basis_state.index(1)][::-1] # to be used for manual calc below. index order is reversed for qiskit
    basis_state = Statevector(basis_state)
    print(f"For basis state {basis_state.data}:")
    QAOA_result = basis_state.evolve(qc)
    print(f"Applying cost hamiltonian exponentiation --> output state = \n{QAOA_result.data}")
    
    #F(z) method
    variables = [compound_state[i] for i in range(len(compound_state))] # u and v
    s = [(-1) ** variable for variable in variables]
    f = c + sum([alpha[i] * s[i] for i in range(len(s))]) + \
    sum([beta[i][j] * s[i] * s[j] for i,j in pairs])
    QUBO_result = np.exp(-1j * gamma * f) * basis_state.data
    print(f"Applying QUBO F(z) exponentiation --> output state = \n{QUBO_result}")
    # if global phase was not applied to the circuit above:
    if np.allclose(QAOA_result * np.exp(-1j * gamma * c), QUBO_result):
        print(f"States differ by a common global phase of i * gamma * c = {-1j * gamma * c}")
    else: 
        print(f"Error - states are not equivalent")
    print()




# In[1]:


print("PART D - TEN QUBIT CIRCUIT CONSTRUCTION\n")

# from module 8, based on moduel 7:
import random
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp
from itertools import combinations, product
from scipy import linalg

cooling_requirement = 100 # kW
bit = [0, 1]

# power coefficients
power = [10, 17, 25, 0, 0, 0, 0, 0, 0, 0]

# cooling-difference coefficients
diff = [50, 75, 100, -1, -2, -4, -8, -16, -32, -64]

total_assignments = list(product(bit, repeat=len(power)))

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

test_lamda = 1
alpha, beta, c, K, h, J = get_QAOAcoefficients(lamda=test_lamda, p=power, d=diff, constraint=cooling_requirement)

# build the HVAC quantum circuit
num_qubits = len(power)
print(f"Number of qubits = {num_qubits}")
print()
pairs = list(combinations(range(num_qubits), 2)) # all index pairs where i < j
qc = QuantumCircuit(num_qubits) # 10-qubit circuit
gamma = 0.2 # cost unitary variational angle
beta_mixer = 0.4 # mixer unitary variational angle
print(f"number of variational parameters = 2 --> gamma = {gamma}, Bmixer = {beta_mixer}")
print()
print("Building circuit...")
print()
print()

qc.h(range(num_qubits)) # prepare each qubit in the |+> state = equal superposition
gate_count = 10
qc.barrier()
for i in range(num_qubits):
    qc.rz(2 * gamma * alpha[i], i) # first step of cost unitary - single-qubit coefficients
    gate_count += 1
for i,j in pairs:
    qc.rzz(2 * gamma * beta[i][j], i, j) # second step of cost unitary - pairwise coefficients
    gate_count += 1
qc.barrier()
for i in range(num_qubits):
    qc.rx(2 * beta_mixer, i)
    gate_count +=1
    
print("Gate counts:")
for gate, count in qc.count_ops().items():
    print(f"{gate}: {count}")
print()

# test qiskit bitstring ordering
test_bitstring = "0000000100"
print(f"String {test_bitstring} reads as {int(test_bitstring, 2)}") # qiskit should read the test_bitstring as 1 x 2^2 = 4

# draw the circuit
qc.draw(output="mpl", fold=-1)


# In[14]:


print("PART E - EXACT EXPECTATION CALCULATION\n")

import math
import numpy as np
import random
from itertools import product

result = Statevector.from_instruction(qc)
probabilities = result.probabilities() # list of squared magnitudes
print(f"Running 1-layer circuit with λ = {test_lamda}, γ = {gamma}, β = {beta_mixer}...")
print()
print(f"Check: total of probabilities over all assignments = {sum(probabilities)}")
print()
def get_QUBOcost(K, h, J, z):
    if len(h) == len(z) == len(J):
        pass
    else:
        print("Error: Invalid QUBO inputs, dimension mismatch.")
    z_pairs = list(combinations(range(len(z)), 2))    
    F = K + sum([h[i] * z[i] for i in range(len(h))]) + sum([J[i][j] * z[i] * z[j] for i,j in z_pairs])
    return F

total_assignments = [assignment[::-1] for assignment in list(product(bit, repeat=num_qubits))]
QUBOcosts = [get_QUBOcost(K, h, J, assignment) for assignment in total_assignments]
expected_cost = np.dot(probabilities, QUBOcosts)
print(f"Expected cost = {expected_cost}")


# In[65]:


print("F - CLASSICAL PARAMETER OPTIMIZATION\n")

from scipy.optimize import minimize
import random
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp
from itertools import combinations, product
from scipy import linalg

cooling_requirement = 100 # kW
bit = [0, 1]

# power coefficients
power = [10, 17, 25, 0, 0, 0, 0, 0, 0, 0]

# cooling-difference coefficients
diff = [50, 75, 100, -1, -2, -4, -8, -16, -32, -64]

# ensure coefficient matrices are same length
if len(power) == len(diff):
    num_qubits = len(power)
else:
    raise ValueError("Power and cooling coefficient matrices are not equal in size --> invalid!")

pairs = list(combinations(range(num_qubits), 2)) # all index pairs where i < j
total_assignments = [assignment[::-1] for assignment in list(product(bit, repeat=num_qubits))]

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

lamda = 1 # fixed for now
alpha, beta, c, K, h, J = get_QAOAcoefficients(lamda=lamda, p=power, d=diff, constraint=cooling_requirement)

# normalize to make optimization easier
s = 10000 # normalization factor
alpha = [alpha[i] / s for i in range(len(alpha))]
beta = beta / s
c = c / s

def get_QUBOcost(K, h, J, z):
    if len(h) == len(z) == len(J):
        pass
    else:
        raise ValueError("Error: Invalid QUBO inputs, dimension mismatch")
    z_pairs = list(combinations(range(len(z)), 2))    
    F = K + sum([h[i] * z[i] for i in range(len(h))]) + sum([J[i][j] * z[i] * z[j] for i,j in z_pairs])
    return F

QUBOcosts = [get_QUBOcost(K, h, J, assignment) for assignment in total_assignments]
uniform_expected_cost = sum(QUBOcosts) / len(QUBOcosts)

def get_expected_cost(angles, num_qubits, alpha, beta): # angles is a list: [gamma, beta_mixer]
    gamma = angles[0]
    beta_mixer = angles[1]
    
    # build quantum circuit
    qc = QuantumCircuit(num_qubits) # 10-qubit circuit
    qc.h(range(num_qubits)) # prepare each qubit in the |+> state = equal superposition
    qc.barrier()
    for i in range(num_qubits):
        qc.rz(2 * gamma * alpha[i], i) # first step of cost unitary - single-qubit coefficients
    for i,j in pairs:
        qc.rzz(2 * gamma * beta[i][j], i, j) # second step of cost unitary - pairwise coefficients
    qc.barrier()
    for i in range(num_qubits):
        qc.rx(2 * beta_mixer, i)

    # run circuit on |0000000000> to get resulting amplitudes for each possible assignment
    result = Statevector.from_instruction(qc)
    probabilities = result.probabilities() # list of squared magnitudes

    # for debugging:
    #print()
    #print("probability of optimal HVAC state:", probabilities[4])
    #print("most probable state = ", total_assignments[np.argmax(probabilities)])
    #print("probability of most probable state:", np.max(probabilities))
    #print()
    
    expected_cost = np.dot(probabilities, QUBOcosts)

    return expected_cost, result, probabilities

starting_points = [
    [0.01, 0.02],
    [0.5, 1],
    [1, 1.5],
    [1.5, 2],
    [2, 2.5],
    [2.5, 3],
    [3, 3.5],
    [3.5, 4],
    [4, 4.5],
    [4.5, 5],
    [5, 5.5],
    [5.5, 6]
]
results = []

print(f"Building QAOA circuit with λ = {lamda}...")
print()
print(f"Expected cost of uniform distribution = {uniform_expected_cost}")
print()

min_cost = float('inf')
optimal_angles_1layer = None
for starting_point in starting_points:
    print(f"Optimizing QAOA objective, starting from γ = {starting_point[0]}, β = {starting_point[1]}...")
    result = minimize(
        lambda angles: get_expected_cost(angles, num_qubits, alpha, beta)[0],
        x0=starting_point,
        method="COBYLA",
        options={"maxiter": 500}
    )
    print(f"Best parameters found: γ = {result.x[0]}, β = {result.x[1]}")
    print(f"Expected cost = {result.fun}")
    print(f"Reduction in cost compared to uniform distribution = \
    1 - {round(result.fun,1)} / {round(uniform_expected_cost,1)} = \
    {round(100 * (1 - result.fun / uniform_expected_cost), 1)}%") 
    print()
    print()
    if result.fun < min_cost:
        min_cost = result.fun
        optimal_angles_1layer = result.x

print(f"Optimal angles for 1 layer circuit: γ = {optimal_angles_1layer[0]}, β = {optimal_angles_1layer[1]}") 


# In[66]:


print("PART G - SAMPLING AND DECODING\n")
print()

import random
import itertools
from collections import Counter

num_samples = 2000

cost, state, probabilities = get_expected_cost(optimal_angles_1layer, num_qubits, alpha, beta)

samples = random.choices(total_assignments, weights=probabilities, k=num_samples)
counts = Counter(samples)
most_common, frequency = counts.most_common(1)[0]
a, b, c, y0, y1, y2, y3, y4, y5, y6 = [most_common[i] for i in range(len(most_common))]
cost_most_common = get_QUBOcost(K, h, J, most_common)
num_feasible = 0
best_feasible_cost = float('inf')
best_feasible = None
for sample in samples:
    # is it feasible?
    if sample[0] * 50 + sample[1] * 75 + sample[2] * 100 >= cooling_requirement:
        num_feasible += 1
    else:
        continue
    sample_cost = get_QUBOcost(K, h, J, sample)   
    if sample_cost <= best_feasible_cost:
        best_feasible_cost = sample_cost
        best_feasible = sample

def status(binary_var):
    if binary_var == 0:
        return "OFF"
    elif binary_var == 1:
        return "ON"
    else: 
        return None

print(f"Running QAOA with γ = {optimal_angles_1layer[0]}, β = {optimal_angles_1layer[1]}...")
print()
print(f"Resulting state vector found. Sampling {num_samples} times...")
print()
print(f"Most common sampled state = {most_common}")
print(f"A {status(a)}, B {status(b)}, C {status(c)}, y0 {status(y0)}, y1 {status(y1)}, y2 {status(y2)},\
 y3 {status(y3)}, y4 {status(y4)}, y5 {status(y5)}, y6 {status(y6)}")
print(f"Cost of most common sample = {cost_most_common}")
print(f"Percent of feasible samples (cooling >= 100 kW) = {num_feasible} / {num_samples} = \
{round(100 * num_feasible / num_samples, 1)}")
print(f"Best feasible sample = {best_feasible} (cost = {best_feasible_cost})")
        


# In[80]:


print("PART H - LAYER DEPTH COMPARISON\n")

from scipy.optimize import minimize
import random
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp
from itertools import combinations, product
from scipy import linalg

cooling_requirement = 100 # kW
bit = [0, 1]

# power coefficients
power = [10, 17, 25, 0, 0, 0, 0, 0, 0, 0]

# cooling-difference coefficients
diff = [50, 75, 100, -1, -2, -4, -8, -16, -32, -64]

# ensure coefficient matrices are same length
if len(power) == len(diff):
    num_qubits = len(power)
else:
    raise ValueError("Power and cooling coefficient matrices are not equal in size --> invalid!")

pairs = list(combinations(range(num_qubits), 2)) # all index pairs where i < j
total_assignments = [assignment[::-1] for assignment in list(product(bit, repeat=num_qubits))]

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

lamda = 1 # fixed for now
alpha, beta, c, K, h, J = get_QAOAcoefficients(lamda=lamda, p=power, d=diff, constraint=cooling_requirement)

# normalize to make optimization easier
s = 10000 # normalization factor
alpha = [alpha[i] / s for i in range(len(alpha))]
beta = beta / s
c = c / s

def get_QUBOcost(K, h, J, z):
    if len(h) == len(z) == len(J):
        pass
    else:
        raise ValueError("Error: Invalid QUBO inputs, dimension mismatch")
    z_pairs = list(combinations(range(len(z)), 2))    
    F = K + sum([h[i] * z[i] for i in range(len(h))]) + sum([J[i][j] * z[i] * z[j] for i,j in z_pairs])
    return F

QUBOcosts = [get_QUBOcost(K, h, J, assignment) for assignment in total_assignments]
uniform_expected_cost = sum(QUBOcosts) / len(QUBOcosts)

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
    result = Statevector.from_instruction(qc)
    probabilities = result.probabilities() # list of squared magnitudes

    # for debugging:
    #print()
    #print("probability of optimal HVAC state:", probabilities[4])
    #print("most probable state = ", total_assignments[np.argmax(probabilities)])
    #print("probability of most probable state:", np.max(probabilities))
    #print()
    
    expected_cost = np.dot(probabilities, QUBOcosts)

    return expected_cost, result, probabilities, qc

# define starting angles for 2-layer and 3-layer optimization

initial_angles = [
    optimal_angles_1layer,
    [0, 0]
]

print(f"Building QAOA circuit with λ = {lamda}...")
print()
print(f"Expected cost of uniform distribution = {uniform_expected_cost}")
print()

num_layers = 5 # tuneable
min_cost = float('inf')
optimal_angles = None
for i in range(1, num_layers): # skip 1-layer trial as that has been calculated already
    print(f"Optimizing QAOA using {i + 1} layers. Starting angles:")
    print()
    for i in range(len(initial_angles)):
        print(f"Layer {i + 1} --> γ = {initial_angles[i][0]}, β = {initial_angles[i][1]}")
    print()
    result = minimize(
        lambda angles: get_expected_cost(angles, num_qubits, alpha, beta)[0],
        x0=np.array(initial_angles).flatten(),
        method="COBYLA",
        options={"maxiter": 3000}
    )
    optimized_angles = result.x.reshape(-1,2) # re-format so angles are listed in pairs
    initial_angles = np.append(optimized_angles, [[0,0]], axis=0)
    print("Best parameters found:")
    for i in range(len(optimized_angles)):
        print(f"    Layer {i + 1} --> γ = {optimized_angles[i][0]}, β = {optimized_angles[i][1]}")
    print(f"Expected cost = {result.fun}")
    print(f"Reduction in cost compared to uniform distribution = \
    1 - {round(result.fun,1)} / {round(uniform_expected_cost,1)} = \
    {round(100 * (1 - result.fun / uniform_expected_cost), 1)}%") 
    print()
    print()
    if result.fun < min_cost:
        min_cost = result.fun
        optimal_angles = result.x.reshape(-1,2)

print("Optimal angles:")
for i in range(len(optimal_angles)):
    print(f"    Layer {i + 1}: γ = {optimal_angles[i][0]}, β = {optimal_angles[i][1]}")



# In[75]:


print("PART I - WRITTEN REFLECTION\n")

print('''cost-dependent phase-assignment is only useful if combined with mixing because phases
don't change the probability of measuring any given HVAC configuration. We apply mixing to 
explore different probabilities for each state, with the goal of getting the optimal state to 
attain a higher probability of being measured.
The ising constant can be dropped from the circuit because it adds the same cost to all
assignments and therefore does not change relative costs for minimization purposes. It should be
added back when reporting the expected cost becuase it is in fact part of the cost.
10 qubits are needed for slack encoding for this problem: 3 for each HVAC unit, and then
7 binary slack variables to cover the highest possible cooling overshoot -- 125 kw.
This experiment does not necessarily prove that using a quantum circuit is better for finding
the optimal solution, since in this case, the optimal solution is easily calculated using 
classical methods. however, it does show that a quantum circuit could theoretically be used for
a much more complicated system, say one involving 1000 HVAC units, where the number of 
possibilities is likely far too high to search through and optimize classically.''')


# In[ ]:





# In[ ]:




