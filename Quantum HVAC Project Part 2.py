# In[40]:


print("PART A: WRITTEN QUANTUM-STATE MAPPING\n")
basis_states = [np.array([1, 0]), np.array([0, 1])]
for state in basis_states:
    if state[0] == 1:
        print("Unit OFF state: |0>")
    elif state[1] == 1:
        print("Unit ON state: |1>")
    else:
        print("Invalid State")
    for value in state:
        print(value)
    print()


# In[76]:


print("PART B: AMPLITUDE CALCUATIONS\n")

import sympy as sp
from IPython.display import display, Math


amplitudes = [np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]),
              np.array([np.sqrt(3) / 2, 1/2]),
              np.array([1/2, -np.sqrt(3)/2])
             ]
quantum_states = []
for amplitude in amplitudes:
    if len(amplitude) != len(basis_states):
        print("Invalid matrix dimensions")
        continue
    quantum_state = amplitude @ basis_states
    quantum_states.append(quantum_state)

sp.init_printing()
for i in range(len(quantum_states)):
    print(f"Qubit #{i+1}:")
    for j in range(len(quantum_states[i])):
        if j == 0:
            unit_state = "OFF"
        if j == 1:
            unit_state = "ON"
        if j > 1:
            print("Invalid state")
            continue
        display(Math(
            rf"({sp.latex(sp.nsimplify(quantum_states[i][j]))})^2 \
                 \text{{ = Probability of unit being {unit_state}}}"
               ))
    print()


# In[102]:


print("PART C: SMALL PYTHON SIMULATION\n")

rng = np.random.default_rng()

for i in range(len(quantum_states)):
    print(f"Qubit #{i + 1}:")
    probabilities = [amplitude ** 2 for amplitude in quantum_states[i]] # this formula will change when complex components are introduced
    euclidean_norm = round(sum(probabilities), 5)
    display(Math(rf"\text{{Euclidean Norm}} = ({sp.latex(sp.nsimplify(quantum_states[i][0]))})^2 + \
                              ({sp.latex(sp.nsimplify(quantum_states[i][1]))})^2 = \
                              {euclidean_norm}"))
    if euclidean_norm == 1:
        print("State is normalized. Sampling...")
    else:
        print("State is NOT normalized")
        continue
    sample_size = 1000
    samples = rng.choice([0,1], size=sample_size, p=probabilities)
    num_zeros = sum(samples == 0)
    num_ones = sum(samples == 1)
    print(f"Unit is OFF {100 * num_zeros / sample_size} % of the time")
    print(f"Unit is ON {100 * num_ones / sample_size} % of the time")
    print()
    print()


# In[104]:


print("PART D: HVAC CONNECTION\n")

print('''FOR A UNIT IN EQUAL SUPERPOSITION BEING MEASURED 1000 TIMES, 
YOU WOULD EXPECT A ZERO = OFF 500 TIMES AND A 1 = ON 500 TIMES.
DOESN'T MEAN UNIT IS AT 50% CAPACITY, 
JUST MEANS THAT IN THIS SUPERPOSITION,
THERE IS A 50-50 CHANCE OF MEASURING "ON" OR "OFF".''')


# In[ ]:




