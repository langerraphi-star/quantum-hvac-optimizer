
# module 6

print("PART A - HAMILTONIAN FUNDAMENTALS\n")

print('''an observable is a matrix representation of a physical quantity that we are interested
in, as opposed to the state of the system. for example, it could represent the total power output
rather than whether a certain unit is on or off. A Hamiltonian is a specific observable that 
represents the total energy of the physical system. 
An eigenstate is a state of the system where the observable (or Hamiltonian), 
if measured, would produce a certain physical outcome with certainty. That value is the 
corresponding eigenvalue. The ground state is the lowest-energy state of the Hamiltonian's
eigenstates, and it has an associated ground-state energy value. The expectation value is
the probability-weighted average value of measuring the observable many times. In the context of
HVAC, the hamiltonian represents the overall "cost" of the system.
H|psi> is not a measurement because it is an algebraic equation. it is simply a matrix
operation that transforms the quantum state; the result is a vector, not a measurement.''')


# In[3]:


print("PART B - ONE-QUBIT WARM-UP \n")

import numpy as np

H = np.diag([2,7])
print(f"H = \n{H}")
print()
basis_states = [ np.array([[1], [0]]), np.array([[0], [1]]) ]
def get_eigenvalue(matrix, vector):
    if np.allclose(vector, 0): # zero vector is not an eigen vector by definition
        return False, "Zero vector cannot be an eigenvector"
    result = matrix @ vector
    non_zero_index = np.argmax(vector != 0) # returns value of first non-zero element in vector
    possible_lambda = result[non_zero_index] / vector[non_zero_index]

    # check if all elements in result satisfy the lambda relationship
    if np.allclose(result, possible_lambda * vector):
        return True, possible_lambda
    else:
        return False, None

for state in basis_states:
    check_eigenvector = get_eigenvalue(H, state)
    check = check_eigenvector[0]
    eigenvalue = check_eigenvector[1]
    print(f"Is {state} \nan eigenvector of \n{H}?")
    print(f"{check}, eigenvalue = {eigenvalue}")
    print()

print()
print('''4.5 is not a possible single-shot outcome because it is not an eigenvalue.
it is the expected value for a qubit in the |+> state, but that is the average over
many measurements.''')
    


# In[24]:


print("PART C: HVAC POWER HAMILTONIAN\n")
from itertools import product
from functools import reduce
import sympy as sp

# top row is cooling capacity, bottom row is power consumption
cooling_requirement = 100 # kW

x = np.array([
        [50, 75, 100],
        [10, 17, 25]
    ])

switch = [0, 1] # one-bit switch representing a single HVAC unit
configurations = list(product(switch, repeat=3)) # all possible 3-unit configurations

states = ["A OFF, B OFF, C OFF", 
          "A OFF, B OFF, C ON", 
          "A OFF, B ON, C OFF", 
          "A OFF, B ON, C ON", 
          "A ON, B OFF, C OFF", 
          "A ON, B OFF, C ON", 
          "A ON, B ON, C OFF", 
          "A ON, B ON, C ON"]

cooling = [(x @ configuration)[0] for configuration in configurations]
consumption = [(x @ configuration)[1] for configuration in configurations]

x_cooling = np.diag(cooling)
x_cooling_t = x_cooling.conj().T

x_consumption = np.diag(consumption)
x_consumption_t = x_consumption.conj().T

def find_eigenvalues(matrix):
    A = sp.Matrix(matrix)
    lamda = sp.symbols('lamda')
    I = sp.eye(A.shape[0])
    char_matrix = A - lamda * I # characteristic matrix
    char_poly = char_matrix.det()
    eigenvalues = sp.solve(char_poly, lamda)
    eigenvectors = []
    for eigenvalue in eigenvalues:
        A_float = np.array(A.tolist(), dtype=float)
        eigenvalue_float = float(eigenvalue) # convert from sympy to numpy format
        I_float = np.array(I.tolist(), dtype=float)
        char_matrix_i = A_float - eigenvalue_float * I_float
        # Decompose the characteristic matrix
        # U: Left singular vectors, S: Singular values, Vt: Right singular vectors (transposed)
        U, S, Vt = np.linalg.svd(char_matrix_i)
        eigenvector = Vt[-1,:] # last row in Vt corresponds to the zero value in S
        eigenvectors.append(eigenvector)
    return eigenvalues, eigenvectors

cooling_eigenvalues, cooling_eigenvectors = find_eigenvalues(x_cooling)
consumption_eigenvalues, consumption_eigenvectors = find_eigenvalues(x_consumption)

print(f"Cooling Operator = \n{x_cooling}")
print()
print(f"Cooling Operator conjugate transpose = \n{x_cooling_t}")
print()
if np.array_equal(x_cooling, x_cooling_t):
    print("Cooling Operator is equal to its own conjugate transpose --> Hermitian")
else:
    print("Cooling Operator is NOT equal to its own conjugate transpose --> NOT Hermitian")
print()
print("Cooling eigenspectrum:")

cooling_gs_energy = float('inf') # gs = ground state
cooling_gs = None
for eigenvalue, eigenvector in zip(cooling_eigenvalues, cooling_eigenvectors):
    print(f"For Eigenvalue λ = {eigenvalue}: Eigenvector = {eigenvector}")
    if eigenvalue < cooling_gs_energy:
        cooling_gs_energy = eigenvalue
        cooling_gs = eigenvector
print()
print("Cooling ground state = ")
for index, element in enumerate(cooling_gs):
    if element == 0:
        continue
    else:
        print(f"{element ** 2 * 100} % chance of: {states[index]}")
print()
print(f"Cooling ground state energy = {cooling_gs_energy} kW")   
if cooling_gs_energy < cooling_requirement:
    print(f"Ground state energy is less than the cooling requirement of {cooling_requirement} kW")
else:
    print(f"Ground state satisfies the cooling requirement of {cooling_requirement} kW")
print()
print()
print(f"Consumption Operator = \n{x_consumption}")
print()
print(f"Consumption Operator conjugate transpose = \n{x_consumption_t}")
print()
if np.array_equal(x_consumption, x_consumption_t):
    print("Consumption Operator is equal to its own conjugate transpose --> Hermitian")
else:
    print("Consumption Operator is NOT equal to its own conjugate transpose --> NOT Hermitian")
print("Consumption eigenspectrum:")
for eigenvalue, eigenvector in zip(consumption_eigenvalues, consumption_eigenvectors):
    print(f"For Eigenvalue λ = {eigenvalue}: Eigenvector = {eigenvector}")
print()
print()



# In[34]:


print("PART D - COOLING FEASiBILITY \n")

import pandas as pd

feasible = [cooling[i] >= cooling_requirement for i in range(len(cooling))]

min_feasible_state = None
min_feasible_consumption = float('inf')
for i in range(len(consumption)):
    if cooling[i] >= cooling_requirement and consumption[i] < min_feasible_consumption:
        min_feasible_consumption = consumption[i]
        min_feasible_state = states[i]

for i in range(len(feasible)):
    if states[i] == min_feasible_state:
        feasible[i] = str(feasible[i]) + " - lowest-consumption feasible state"

df = pd.DataFrame({
    "State": states,
    "Cooling (kW)":     cooling,
    "Consumption (kW)": consumption,
    "Feasible":         feasible
})

df = df.style.set_properties(**{'text-align': 'center'}).set_table_styles\
([{'selector': 'th', 'props': [('text-align', 'center')]} ])
df


# In[126]:


print("PART E - PENALTY HAMILTONIAN\n")

def cost_hamiltonian(λ):
    cost = [consumption[i] + λ * max(0, 100 - cooling[i]) ** 2 for i in range(len(consumption))]
    x_cost = np.diag(cost)
    print(f"Cost Hamiltonian with λ = {λ}: \n{x_cost}")
    print()
    for i, val in enumerate(cost):
        if val == min(cost):
            cost_gs_energy = val
            cost_gs = states[i]
    
    print(f"Cost ground state = {cost_gs}")
    print(f"Cost ground state energy = {cost_gs_energy}")
    if cost_gs == min_feasible_state:
        print(f"--> Hamiltonian with λ = {λ} correctly produces ground state equal to minimum feasible from manual calculation.")
    else:
        print(f"--> Hamiltonian with λ = {λ} does NOT produces ground state equal to minimum feasible from manual calculation. Higher λ needed.")
    
    return cost, x_cost
    
eigenvalues_1, H_cost_1 = cost_hamiltonian(λ=1)
print()
print()
eigenvalues_01, H_cost_01 = cost_hamiltonian(λ=0.01)


# In[76]:


print("PART F - EIGENSTATE CHECKS\n")

import random 
import itertools
from functools import reduce

basis = [np.array([ [1], [0] ]) , np.array([ [0], [1] ])]
basis_combinations = list(itertools.product(basis, repeat = 3))
state_vectors = [reduce(np.kron, combination) for combination in basis_combinations]

num_checks = 3
random_idxs = random.sample(list(range(len(states))), k=num_checks)
random_states = [states[random_idxs[i]] for i in range(num_checks)]
random_vectors = [state_vectors[random_idxs[i]] for i in range(num_checks)]
random_eigenvalues = [eigenvalues_1[random_idxs[i]] for i in range(num_checks)]

for i in range(num_checks):
    print(f"Checking state {random_states[i]}...")
    print()
    hamiltonian_check = H_cost_1 @ random_vectors[i]
    eigenvalue_check = random_eigenvalues[i] * random_vectors[i]
    print("Hamiltonian check:")
    print(f"{H_cost_1} \nX \n{random_vectors[i]} \n= \n{hamiltonian_check}")
    print()
    print("Eigenvalue check:")
    print(f"{random_eigenvalues[i]} \nX \n{random_vectors[i]} \n= \n{eigenvalue_check}")
    if np.array_equal(hamiltonian_check, eigenvalue_check):
        print("--> Eigenstate verified")
    else:
        print("--> Invalid eigenstate")

    print()
    print()



# In[127]:


print("PART G - SUPERPOSITION EXPECTATIONS \n")

import math

angle = random.uniform(0, 2 * math.pi)
amplitudes = [np.sin(angle), np.cos(angle)]

# si = superimposition
si_idxs = random.sample(list(range(len(states))), k=len(amplitudes))
si_states = [states[si_idxs[i]] for i in range(len(amplitudes))]
si_vectors = [state_vectors[si_idxs[i]] for i in range(len(amplitudes))]

ket_superposition = np.array([ [0] for i in range(len(si_vectors[0]))])
for i in range(len(amplitudes)):
    ket_superposition = ket_superposition + amplitudes[i] * si_vectors[i]
bra_superposition = ket_superposition.conj().T
hamiltonian_cost = (bra_superposition @ H_cost_1 @ ket_superposition).flatten()
manual_cost = sum([abs(ket_superposition[i][0]) ** 2 * eigenvalues_1[i] for i in range(len(ket_superposition))])

print(f"Superposition = \n {amplitudes[0]} X \n{si_vectors[0]} \n + \n{amplitudes[1]} X \n{si_vectors[1]} \n= \n{ket_superposition}")
print()
print(f"Hamiltonian cost = \n{bra_superposition} \nX\n {H_cost_1} \nX\n{ket_superposition} \n=\n{hamiltonian_cost[0]}")
print()
print()
print("Manually-calculated cost = ")
counter = 0
for i in range(len(ket_superposition)):
    if ket_superposition[i][0] != 0:
        counter += 1
        print(f"{ket_superposition[i][0]}^2 X {eigenvalues_1[i]} = {abs(ket_superposition[i][0]) ** 2 * eigenvalues_1[i]}")
        if counter != 2:
            print("+")
        else:
            print("=")
    else:
        continue
print(manual_cost)
print()
if np.isclose(hamiltonian_cost, manual_cost):
    print("Cost values are equal --> hamiltonian verified!")
else:
    print("Cost values are not equal --> invalid hamiltonian")

print()
print(f"For comparison - ground state cost = {min(eigenvalues_1)}")
print()
print()
print("PART H - SAMPLING\n")

num_samples = 1000
choices = list(range(len(states))) # numbers 0 thru 7, representing each possible HVAC state
samples = random.choices(choices, weights=[abs(element[0])**2 for element in ket_superposition], k=num_samples)
costs = []
feasibles = 0
for sample in samples: # each sample is an index
    if feasible[sample] != False:
        feasibles += 1
    ket = state_vectors[sample]
    bra = ket.conj().T
    cost_i = bra @ H_cost_1 @ ket
    costs.append(cost_i)

avg_cost = (sum(costs) / len(costs)).flatten()
print(f"Taking {num_samples} samples...")
print(f"Average cost from {num_samples} samples: {avg_cost[0]}")
print(f"Exact cost value (from above): {hamiltonian_cost[0]}")
print(f"Fraction of feasible samples = {feasibles} / {num_samples} = {feasibles / num_samples}")


# In[128]:


print("PART I - WRITTEN REFLECTION\n")

print('''Minimizing power alone is insufficient because it does not account for states that
fail to meet the cooling requirement. the penalty adds an extra cost to states that fail the
requirement. if the penalty is sufficiently large, the overall cost of the failed states
will always be more than states that meet the requirement, even if the latter has a higher 
consumption value. as a result, we can find the optimal solution by minimizing a single function, 
i.e. the cost function.
A low expectation value means we expect the average cost to be low over infinitely-many samples.
however, any individual measurement may yield a high-cost state in theory.
A cost hamiltonian represents an algebraic value associated with each particular state, rather 
than a physical quantity that emerges from that state (like kinetic energy, for example).''')


# In[ ]:




