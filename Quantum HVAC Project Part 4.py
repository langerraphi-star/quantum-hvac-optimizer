
# module 4

import numpy as np
from itertools import product
from functools import reduce

print("PART A: BUILD THE MULTI-QUBIT BASIS\nand\nPART B: MAP THE HVAC CONFIGURATIONS\n")

single_positions = [0, 1] # positions a single bit can take
single = {"ket_0 = A OFF": (np.array([[1], [0]]), (0)),
          "ket_1 = A ON": (np.array([[0], [1]]), (1))
                }

double_positions = list(product(single_positions, repeat=2)) # produces tuples of 2-bit combinations, 
                                                             # e.g (1, 0)
double_combinations = list(product([value[0] for value in single.values()], repeat=2)) # produces tuples of 2 single qubits to be tensored,
                                                                                       # e.g (ket_1, ket_1)
triple_positions = list(product(single_positions, repeat=3))
triple_combinations = list(product([value[0] for value in single.values()], repeat=3))

double = {"ket_00 = A OFF, B OFF": None,
          "ket_01 = A OFF, B ON": None,
          "ket_10 = A ON, B OFF": None,
          "ket_11 = A ON, B ON": None
        }

triple = {"ket_000 = A OFF, B OFF, C OFF": None,
          "ket_001 = A OFF, B OFF, C ON": None,
          "ket_010 = A OFF, B ON, C OFF": None,
          "ket_011 = A OFF, B ON, C ON": None,
          "ket_100 = A ON, B OFF, C OFF": None,
          "ket_101 = A ON, B OFF, C ON": None,
          "ket_110 = A ON, B ON, C OFF": None,
          "ket_111 = A ON, B ON, C ON": None
        }

for i in range(len(double_combinations)):
    double_key = list(double.keys())[i]
    double[double_key] = (reduce(np.kron, double_combinations[i]), double_positions[i])

for i in range(len(triple_combinations)):
    triple_key = list(triple.keys())[i]
    triple[triple_key] = (reduce(np.kron, triple_combinations[i]), triple_positions[i])

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

    


# In[211]:


print("PART C: REPRODUCE THE CLASSICAL OPTIMUM\n")

# top row is cooling capacity, bottom row is power consumption
x = np.array([
        [50, 75, 100],
        [10, 17, 25]
    ])

success_states = {}
for state_name, state in triple.items():
    print(f"{state_name}:")
    power = x @ (np.array(state[1]).reshape(-1,1)) # convert bit tuple into column vector and multiply
    cooling = power[0][0]
    consumption = power[1][0]
    check = cooling >= cooling_demand
    print(f"Power Consumption = {consumption} kW")
    print(f"Cooling Output = {cooling} kW")
    print(f"Cooling Demand = {cooling_demand} kW")
    print(f"Demand met? {check}")
    print()
    if check:
        success_states[state_name] = (cooling, consumption)

best_state = None
lowest_consumption = float('inf')
for state, result in success_states.items():
    if result[1] <= lowest_consumption:
        lowest_consumption = result[1]
        best_state = state
print()
print(f"Best State: {best_state}")
print(f"Cooling Output: {success_states[best_state][0]} kW")
print(f"Power Consumption: {lowest_consumption} kW")


# In[220]:


print("PART D: SINGLE-QUBIT GATES IN A LARGER SYSTEM\n")

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

triple_gates = {
    "XII": np.kron(single_gates["X"], np.kron(single_gates["I"], single_gates["I"])),
    "IXI": np.kron(single_gates["I"], np.kron(single_gates["X"], single_gates["I"])),
    "IIX": np.kron(single_gates["I"], np.kron(single_gates["I"], single_gates["X"]))
}

for state_name, state in triple.items():
    print(f"Transforming {state_name}:")
    print()
    for gate_name, gate in triple_gates.items():
        result = gate @ state[0]
        print(f"Applying {gate_name} --> result = \n{result}")
        for key, value in triple.items():
            if np.array_equal(result, value[0]):
                new_state = key
                print(f"New state: {new_state}")
        print()
    print()
    print()
    
    
    


# In[227]:


print("PART E: CNOT\n")
print("CNOT = controlled NOT. If the control bit is 1, flip the target bit.")
print("The first bit is the control, the second is the target.")
print()

inputs = ["ket_00", "ket_01", "ket_10", "ket_11"]
outputs = ["ket_00", "ket_01", "ket_11", "ket_10"]
print(f"{'Input':<10} {'Expected Output':<15}")
for input_state, output_state in zip(inputs, outputs):
    print(f"{input_state:<10} {output_state:<15}")
print() 

CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

for state_name, state in double.items():
    input_state_index = inputs.index(state_name.split(" = ")[0])
    result = CNOT @ state[0]
    print(f"Applying CNOT to {state_name}")
    print(f"Result = \n{result}")
    for key, value in double.items():
        if np.array_equal(result, value[0]):
            new_state = key
            print(f"New state: {new_state}")
            if new_state.split(" = ")[0] == outputs[input_state_index]:
                print("Truth Table validated!")
    print()
    


# In[250]:


print("PART F: CREATE AN ENTANGLED STATE\n")

HI = np.kron(single_gates["H"], single_gates["I"]) # applies H to first qubit but not second

ebits = {
    "ebit1": ("HI", HI @ list(double.values())[0][0]), # applies HI to ket_00
    "ebit2": ("HI then CNOT", CNOT @ (HI @ list(double.values())[0][0])) # applies HI to ket_00
}

#ebits_factored = {}
for ebit,state in ebits.items():
    # convert state to symbolic expression
    state_sym = sp.Matrix([
        [sp.nsimplify(x[0])] for x in state[1]
    ])
    #print(state_sym)
    factor = reduce(sp.gcd, [element[0] for element in state_sym.tolist()])
    #print(factor)
    ebit_factored = np.array([[sp.nsimplify(element[0] * factor)] for element in state[1]])
    ket_factors = []
    for i in range(len(ebit_factored)):
        if ebit_factored[i][0] == 1:
            ket_factors.append(list(double.keys())[i].split(" = ")[0])
    print(f"{ebit}: {state[0]} gate applied to ket_00")
    print(f"Result = \n{state[1]} \n = \n{1 / factor} x \n{ebit_factored}")
    print("=")
    for i in range(len(ket_factors)):
        print(f"{1 / factor} x {ket_factors[i]}")
        if i != len(ket_factors) - 1: # not last element in list
            print("+")
    print()
    probabilities = [round(abs(element[0]) ** 2,5) for element in state[1]]
    for i in range(len(state[1])):
        print(f"Probability of {double_positions[i]} = {probabilities[i]}")
    print(f"Sum of probabilities = {round(sum(probabilities),5)}")
    if round(sum(probabilities),5) == 1:
        print("Normalization verified!")
    else: 
        print("Un-normalized - invalid state")
    print()
        
    


# In[254]:


print("PART G - COMPARE SUPERPOSITION AND ENTANGLEMENT \n")

print('''(1/2) x (ket_00 + ket_01 + ket_10 + ket_11) can be factored,
so the two bits are still independent of each other.
(1/sqrt(2)) x (ket_00 + ket_11) cannot be factored, 
and therefore the two bits are not independent of each other, i.e. they are entangled and depend on each other.
''')


# In[255]:

print("PART H - HVAC REFLECTION \n")

print('''each HVAC unit can be either OFF or ON, i.e. 0 or 1.
so with 3 units, there are 8 possible configuration.
each configuration is represented by one dimension in the quantum state space.
by using one quantum state to represent the amplitudes / probabilities of each configuration,
we now have a mathematical state space in which later quantum operations 
can transform amplitudes associated with different candidate HVAC configurations.
''')


# In[ ]:




