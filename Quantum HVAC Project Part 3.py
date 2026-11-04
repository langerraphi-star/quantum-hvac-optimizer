
# In[156]:


# module 3
print("PART A: DEFINE THE GATES\n")

basis_states = {"ket_0": np.array([[1], 
                                   [0]]),
                "ket_1": np.array([[0], 
                                   [1]])
               }
gates = {
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
    ])
}

for state, vector in basis_states.items():
    print(f"{state} = {vector.flatten()}")

print()
for gate, matrix in gates.items():
    print(f"{gate} = {matrix}")


# In[168]:


print("PART B: BASIS STATE TRANSFORMATIONS\n")

for state,vector in basis_states.items():
    print(f"Transforming {state} = {vector.flatten()}:")
    print()
    for gate, matrix in gates.items():
        new_state = matrix @ vector
        print("Applying ", f"{gate} --> {new_state.flatten()}")
        probabilities = [round(abs(value[0]) ** 2,5) for value in new_state]
        print(f"Probability of measuring OFF = {probabilities[0]}") 
        print(f"Probability of measuring ON = {probabilities[1]}")
        if sum(probabilities) == 1:
            print("Probabilities sum to 1 --> normalized")
        else:
            print("Probabilities do NOT sum to 1 --> NOT normalized")
        print()
    print()
    print()


# In[161]:


print("PART C: HVAC INTERPRETATION\n")

print('''for ket-0, the unit starts out as definitely being OFF.
if you apply the X gate, it becomes definitely ON. You would measure ON with certainty.
if you apply the Z gate, nothing changes. You would still measure OFF with certainty.
if you apply the Y gate, it becomes definitely ON (because |i|^2 = 1) with the phase having changed from 1 to i. You would measure ON with certainty.
if you apply the H gate, the unit now has a 50-50 chance of being measured ON or OFF.''')


# In[158]:


print("PART D: BUILD GATE SEQUENCES\n")

circuits = {"XX (X then X)": gates["X"] @ gates["X"],
            "HH (H then H)": gates["H"] @ gates["H"],
            "HZH (H then Z then H)": gates["H"] @ gates["Z"] @ gates["H"],
            "XH (H then X)": gates["X"] @ gates["H"]
           }

rng = np.random.default_rng()
sample_size = 1000

for circuit, matrix in circuits.items():
    result = matrix @ basis_states["ket_0"]
    print(f"Applying {circuit} to ket_0 = [1, 0] --> final state vector: {result.flatten()}")
    probabilities = [round(abs(value) ** 2,5) for value in result.flatten()]
    print(f"Probability of measuring OFF = {probabilities[0]}")
    print(f"Probability of measuring ON = {probabilities[1]}")
    
#print("PART E: SAMPLING EXPERIMENT\n")
    print("Sampling...")
    samples = rng.choice([0,1], size=sample_size, p=probabilities)
    num_zeros = sum(samples == 0)
    num_ones = sum(samples == 1)
    print(f"Unit is measured OFF {100 * num_zeros / sample_size} % of the time")
    print(f"Unit is measured ON {100 * num_ones / sample_size} % of the time")
    print()

    
            


# In[159]:


print("PART F: SHORT WRITTEN REFLECTION\n")

print('''
(1)Even though Z does not directly change the measurement probabilities,
it matters because it causes a phase change, which can lead to 
altered measurement probabilities further on if the circuit involves 
other gates. As we can see, applying HH does not yield the same result 
as applying HZH. So clearly, Z makes a difference.
(2) H helps move the system from a purely classical model because
it can generate a 50-50 ON/OFF state from a state that was previously
either fully ON or fully OFF.
(3) Applying the X = NOT gate twice yields the original state. This 
tells us that certain gates can be reversed if applied twice. However, this does not 
necessarily apply to all gates. ''')






# In[ ]:




