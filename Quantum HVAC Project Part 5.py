

# In[34]:


# module 5
import numpy as np
import sympy as sp

print("PART A - BRA-KET CALCULATIONS\n")

basis_states = [ np.array([[1],[0]]), np.array([[0],[1]]) ]
amplitudes = [
         [1/np.sqrt(2), 1j/np.sqrt(2)],
         [np.sqrt(3)/2, -1j/2],
         [2j/np.sqrt(5), -1/np.sqrt(5)],
]
print("Amplitudes:")
for amplitude in amplitudes:
    print([sp.nsimplify(value) for value in amplitude])
print()
print()
for i in range(len(amplitudes)):
    ket = sum([amplitudes[i][j] * basis_states[j] for j in range(len(basis_states))])
    #print(ket)
    bra = ket.conj().T
    inner_product = complex(round((bra @ ket).item().real, 2), (bra @ ket).item().imag)
    normalized = round(sum([abs(amplitude) **2 for amplitude in amplitudes[i]]),5) == 1
    print(f"Qubit #{i+1}")
    print()
    print(f"Ket =\n{[[sp.nsimplify(value[0])] for value in ket]}")
    print()
    print(f"Bra = \n{[sp.nsimplify(value) for value in bra[0]]}")
    print()
    print(f"Inner Product = {inner_product}")
    print()
    if normalized:
        print(f"({sp.nsimplify(abs(amplitudes[i][0]))})^2 + ({sp.nsimplify(abs(amplitudes[i][1]))})^2 = 1 --> Normalized!")
    else:
        print((f"({sp.nsimplify(amplitudes[i][0])})^2 + ({sp.nsimplify(amplitudes[i][1])})^2 != 1 --> Not normalized!"))
    print()
    print()
    
    



# In[42]:


print('PART B - Z EXPECTATION VALUES \n')

# from module 3:
single_gates = {
    "X": np.array([
        [0, 1],
        [1, 0]
    ]),
    "Z": np.array([
        [1, 0],
        [0, -1]
    ]),
    "Y": np.array([
        [0, -1j],
        [1j, 0]
    ]),
    "H": (1 / np.sqrt(2)) * np.array([
        [1, 1],
        [1, -1]
    ]),
    "I": np.array([[1, 0],
                  [0, 1]
    ])
}

def expected_values(observable, kets): # observable is a hermitian operator and kets is a list of 2x1 np arrays
    bras = [ket.conj().T for ket in kets]
    expectations = [bras[i] @ observable @ kets[i] for i in range(len(kets))]
    return expectations

gate = "Z"
observable = single_gates[gate]
kets = {"0>": np.array([[1], [0]]), 
        "1>": np.array([[0], [1]]), 
        "+>": (1 / np.sqrt(2)) * np.array([[1], [1]]),
        "->": (1 / np.sqrt(2)) * np.array([[1], [-1]])
       }
expectations = expected_values(observable, list(kets.values()))
for i in range(len(expectations)):
    print(f"Expected value of {gate} in {list(kets.keys())[i]} state: {expectations[i]}")


# In[145]:


from itertools import product
from functools import reduce

print("PART C: BUILD THE HVAC POWER OPERATOR\nand\nPART D: BUILD THE HVAC COOLING OPERATOR\n")

single_positions = [0, 1] # positions a single bit can take
single = {"|0> = A OFF": (np.array([[1], [0]]), (0)),
          "|1> = A ON": (np.array([[0], [1]]), (1))
                }

double_positions = list(product(single_positions, repeat=2)) # produces tuples of 2-bit combinations, 
                                                             # e.g (1, 0)
double_combinations = list(product([value[0] for value in single.values()], repeat=2)) # produces tuples of 2 single qubits to be tensored,
                                                                                       # e.g (ket_1, ket_1)
triple_positions = list(product(single_positions, repeat=3))
triple_combinations = list(product([value[0] for value in single.values()], repeat=3))

double = {"|00> = A OFF, B OFF": None,
          "|01> = A OFF, B ON": None,
          "|10> = A ON, B OFF": None,
          "|11> = A ON, B ON": None
        }

triple = {"|000> = A OFF, B OFF, C OFF": None,
          "|001> = A OFF, B OFF, C ON": None,
          "|010> = A OFF, B ON, C OFF": None,
          "|011> = A OFF, B ON, C ON": None,
          "|100> = A ON, B OFF, C OFF": None,
          "|101> = A ON, B OFF, C ON": None,
          "|110> = A ON, B ON, C OFF": None,
          "|111> = A ON, B ON, C ON": None
        }

for i in range(len(double_combinations)):
    double_key = list(double.keys())[i]
    double[double_key] = (reduce(np.kron, double_combinations[i]), double_positions[i])

for i in range(len(triple_combinations)):
    triple_key = list(triple.keys())[i]
    triple[triple_key] = (reduce(np.kron, triple_combinations[i]), triple_positions[i])

'''
print("Single qubit states:\n")
for state, vector in single.items():
    print(f"{state}:")
    for element in vector:
        print(element)
    print()
    
print("Double qubit states:\n")
for state, vector in double.items():
    print(f"{state}:")
    for element in vector:
        print(element)
    print()

print()
print("Triple qubit states:\n")
for state, vector in triple.items():
    print(f"{state}:")
    for element in vector:
        print(element)
    print()
    
'''

# top row is cooling capacity, bottom row is power consumption
x = np.array([
        [50, 75, 100],
        [10, 17, 25]
    ])

configurations = [np.array(list(triple.values())[i][1]).reshape(-1, 1) for i in range(len(triple))]

consumption = []
cooling = []

def get_energy(equipment_data, configuration):# configuration needs to be an np array in the form [[1],[1],[1]] (e.g.)
    energy = equipment_data @ configuration
    return energy

for i in range(len(configurations)):
    energy = get_energy(x, configurations[i])
    cooling_i = energy[0][0]
    consumption_i = energy[1][0]

    cooling.append(cooling_i)
    consumption.append(consumption_i)

    print(f"Configuration {list(triple.keys())[i]}:\nCooling = {cooling_i} kW \n\
Consumption = {consumption_i} kW")
    print()

x_consumption = np.diag(consumption)
x_cooling = np.diag(cooling)
print(f"Consumption Operator = \n{x_consumption}")
print()
print(f"Cooling Operator = \n{x_cooling}")


# In[200]:


print("PART E - TEST DEFINITE HVAC STATES\n")

import random

test_configs = random.sample([value[0] for value in list(triple.values())],3)
if any(np.array_equal(list(triple.values())[1][0], arr) for arr in test_configs):
    pass
else:
    test_configs.pop()
    test_configs.append(list(triple.values())[1][0])
#print(test_configs)
def get_expectation(observable, ket): 
    bra = ket.conj().T
    expectation = (bra @ observable @ ket).item()
    return expectation

consumption_expectations = [get_expectation(x_consumption, config) for config in test_configs]
cooling_expectations = [get_expectation(x_cooling, config) for config in test_configs]
for i in range(len(test_configs)):
    state_index = next(j for j,v in enumerate(triple.values()) if np.array_equal(v[0],test_configs[i]))
    classical_cooling = cooling[state_index]
    classical_consumption = consumption[state_index]
    quantum_cooling = cooling_expectations[i]
    quantum_consumption = consumption_expectations[i]
    print(f"When system is in {list(triple.keys())[state_index]}:")
    print()
    print(f"Quantum Expected Consumption = {quantum_consumption} kW")
    print(f"Quantum Expected Cooling = {quantum_cooling} kW")
    print()
    print(f"Classical Consumption = {classical_consumption} kW")
    print(f"Classical Cooling = {classical_cooling} kW")
    if classical_cooling == quantum_cooling and classical_consumption == quantum_consumption:
        print("Values match!")
    if classical_cooling != quantum_cooling:
        print("Cooling Mismatch!")
    if classical_consumption != quantum_consumption:
        print("Consumption Mismatch!")
    print()
    print()
        


# In[196]:


print("PART F - TEST A SUPERPOSITION\n")

basis_states = [ np.array([[1], [0]]), np.array([[0], [1]]) ]
num_qubits = 3
amplitudes = []
for i in range(num_qubits):
    # use identity sin^2 + cos^2 = 1 to get a normalized state
    theta = np.random.uniform(0, 2 * np.pi)
    a = np.cos(theta)
    b = np.sin(theta)
    amplitudes.append([a,b])
#print(amplitudes)
#construct the 3-qubit state
qubits = [sum([amplitude[i] * 1j**i * basis_states[i] for i in range(len(basis_states))]).tolist() for amplitude in amplitudes]
#print(qubits)
state = reduce(np.kron, qubits)
print(f"3-Qubit State = \n{state}")
#print(sum([value[0]**2 for value in state])) # normalization check
print()
classical_consumption = 0
classical_cooling = 0
probabilities = []
for i in range(len(state)):
    config = list(triple.items())[i]
    classical_state = np.array(config[1][1]).reshape(-1,1)
    p = abs(state[i][0]) ** 2 # probability of the given configuration
    probabilities.append(p)
    print(f"Probability of {config[0]}: {abs(state[i])}^2 = {round(p,3)}")
    print()
    classical_consumption += p * get_energy(x, classical_state)[1][0]
    classical_cooling += p * get_energy(x, classical_state)[0][0]

classical_consumption = round(classical_consumption, 3)
classical_cooling = round(classical_cooling, 3)
quantum_consumption = round(np.real_if_close(get_expectation(x_consumption, state)).item(), 3)
quantum_cooling = round(np.real_if_close(get_expectation(x_cooling, state)).item(), 3)
print(f"Classical Consumption = {classical_consumption} kW")
print(f"Classical Cooling = {classical_cooling} kW")
print()
print(f"Quantum Consumption = {quantum_consumption} kW")
print(f"Quantum Cooling = {quantum_cooling} kW")
print()
if classical_cooling == quantum_cooling and classical_consumption == quantum_consumption:
    print("Values match!")
if classical_cooling != quantum_cooling:
    print("Cooling Mismatch!")
if classical_consumption != quantum_consumption:
    print("Consumption Mismatch!")

print()
print()
print("PART G - SAMPLING INTERPRETATION\n")

import random

configs = list(triple.keys())

num_samples = 1000

print(f"Taking {num_samples} measurements...")
measurements = random.choices(configs, probabilities, k=num_samples)
print()
#print(measurements[1])
total_consumption = 0
total_cooling = 0
for measurement in measurements:
    state_index = next(j for j,v in enumerate(triple.keys()) if v == measurement)
    sample_consumption = consumption[state_index]
    sample_cooling = cooling[state_index]
    total_consumption += sample_consumption
    total_cooling += sample_cooling

avg_consumption = total_consumption / num_samples
avg_cooling = total_cooling / num_samples

print(f"Average Consumption = {avg_consumption} kW")
print(f"Average Cooling = {avg_cooling} kW")
print()
print(f"Quantum-Expected Consumption = {quantum_consumption} kW")
print(f"Quantum-Expected Cooling = {quantum_cooling} kW")


# In[189]:


print("PART H - WRITTEN REFLECTION\n")

print('''(1) a ket is a vector representing a quantum state.
bra is the conjugate transpose of that vector.''')

print('''(2) "a basis state is an eigenstate of a diagonal operator" means that
when the operator is measured in that basis state, the physical state associated
with that basis state is certain to be the measurement outcome. so if the basis state is |010>
and the operator is the cooling operator, then when you measure the cooling output of the system,
you will with certainty get the cooling associated with A=OFF, B=ON, C=OFF. that cooling output
is the eigenvalue of this eigenstate.''')

print('''(3) a superimposed quantum state has varying probabilties of being in any given 
physical confiuguration. the expected value is the average physical result (e.g. cooling) you
would get if you measured the system an infinite number of times. But any one measurement
can only give you the cooling associated with one particular configuration, which may or may not 
be equal to the expected / overall cooling value.''')

print('''(4) the diagonal value allows you to calculate expected physical quantities for an
HVAC state that is in a superimposed position with varying probabilities. 
So instead of having to go through each possible configuration 
and multiple the corresponding basis state vector by the X matrix and then multiply 
by the associated probability of that configuration and then sum it all up, you just
take your quantum state and your diagonal matrix, do bra @ diag @ ket, and you get your
expected value.''')



# In[ ]:





# In[ ]:




