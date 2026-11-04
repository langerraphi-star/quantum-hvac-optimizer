
# In[18]:


# module 10

print("PART A - ESTABLISH A REPRODUCIBLE BASELINE\n")

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

# define starting angles for 2-layer and 3-layer optimization

initial_angles = [
    [0, 0]
]

print(f"Building QAOA circuit with λ = {lamda}...")
print()
print(f"Expected cost of uniform distribution = {uniform_expected_cost}")
print()

num_layers = 5 # tuneable
min_cost = float('inf')
optimal_angles = None

circuit_data = [] # will be a list of dictionaries

for i in range(num_layers): 
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
    if result.fun < min_cost:
        min_cost = result.fun
        optimal_angles = result.x.reshape(-1,2)

    # run circuit at optimal parameters to retrieve data
    expected_cost, state_result, probabilities, p_optimal, most_likely, prob_feasible, qc = \
get_expected_cost(result.x, num_qubits, alpha, beta)
    circuit_data_i = {
        "Expected Cost": expected_cost, 
        "Resulting State Vector": state_result,
        "Probability Distribution": probabilities,
        "Probability of Optimal State (A OFF B OFF C ON)": p_optimal,
        "Most Likely State to be Measured": most_likely, # this is a tuple: (state, probability)
        "Probability of a Feasible HVAC State (cooling >= 100)": prob_feasible,
        "Quantum Circuit": qc
    }
    circuit_data.append(circuit_data_i)

    for key, value in list(circuit_data_i.items())[3:6]: # print p_optimal, most_likely, prob_feasible
        print(key, ": ", value)

    print()
    print()

print()
print()
                                           
print("Optimal angles:")
for i in range(len(optimal_angles)):
    print(f"    Layer {i + 1}: γ = {optimal_angles[i][0]}, β = {optimal_angles[i][1]}")




# In[28]:


print("PART B - FINITE SHOT SAMPLING\n")

from qiskit_aer import AerSimulator

# use 1-layer circuit to start with
measured_qc = circuit_data[0]["Quantum Circuit"].copy() # creates a copy to avoid impacting the original
measured_qc.measure_all() # measures all 10 qubits into classical bits -- this is added to the circuit

simulator = AerSimulator(seed_simulator=42) # initialize the simulator with a fixed seed for consistent results
N = [100, 500, 2000, 10000] # number of shots per trial
for n in N:
    print(f"Simulating and meausuring 1-layer circuit with {n} shots...")
    print()
    job = simulator.run(measured_qc, shots=n)
    result = job.result()
    counts = result.get_counts() # returns a dictionary -- |ket>: num_times_measured
    freq_optimal = counts.get('0000000100') # A OFF, B OFF, C ON, all slack bits OFF
    cost = 0
    num_feasible = 0
    # print("Empirical outcome frequencies:") # uncomment the code only if the print statement below is uncommented
    for bitstring, frequency in counts.items():
        # display key and value
        # print(f"    {bitstring}: {frequency}") # can uncomment the code but it will be long print

        # get cost
        assignment = [int(digit) for digit in bitstring][::-1]
        cost += frequency * get_QUBOcost(K, h, J, assignment)

        # determine if outcome is feasible (cooling >= 100 kW)
        if int(bitstring[-1]) * 50 + int(bitstring[-2]) * 75 + int(bitstring[-3]) * 100 >= 100:
            num_feasible += frequency
    avg_cost = cost / n
    prob_feasible = f"{round(100 * num_feasible / n, 5)}%"
    
    print(f"Sample-average QUBO cost = {avg_cost}")
    print(f"Frequency of Optimal (A OFF, B OFF, C ON, slack bits OFF) = {freq_optimal}")
    print(f"Fraction of feasible samples (cooling >=100) = {num_feasible} / {n} = {prob_feasible}")
    print()
    print()


# In[38]:


print("PART C - QUANTIFY SHOT UNCERTAINTY\n")

from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

# use 1-layer circuit to start with
measured_qc = circuit_data[0]["Quantum Circuit"].copy() # creates a copy to avoid impacting the original
measured_qc.measure_all() # measures all 10 qubits into classical bits -- this is added to the circuit

simulator = AerSimulator() # initialize the simulator without a fixed seed for independent results
N = [100, 500, 2000, 10000] # number of shots per trial
repeat = 20 # number of times to repeat each experiment
total_avg_costs = []
total_std_devs = [] # standard deviations of each 20-trial run
for n in N:
    print(f"Running 1-layer circuit with {n} shots, repeating {repeat} times and calculating average cost...")
    print()
    cost = 0
    run_avg_costs = [] # will get one value per run = 20 values total
    run_SEs = [] # same, but for standard error of the mean
    for i in range(repeat):
        counts = simulator.run(measured_qc, shots=n).result().get_counts() # returns a dictionary -- |ket>: num_times_measured
        shot_costs = [
            get_QUBOcost(K, h, J, [int(digit) for digit in bitstring][::-1]) 
            for bitstring, frequency in counts.items()
            for _ in range(frequency)
        ]
        run_avg_cost = sum(shot_costs) / n # average cost among all samples measured THIS run
        run_avg_costs.append(run_avg_cost)
        run_SE = np.std(shot_costs , ddof=1) / np.sqrt(n) # SE among all samples measured THIS run
        run_SEs.append(run_SE)
    total_avg_cost = round(sum(run_avg_costs) / repeat, 1) # average cost across all 20 runs for this shot-count
    total_std_dev = round(np.std(run_costs), 1) # same, for SD
    
    total_avg_costs.append(total_avg_cost)
    total_std_devs.append(total_std_dev)
    print(f"Average cost across {repeat} repeats = {total_avg_cost}")
    print(f"Standard Deviation of average costs across {repeat} repeats = {total_std_dev}")
    print(f"Average across {repeat} repeats of Standard Error of {n} shot costs = {round(sum(run_SEs) / repeat, 1)}") 
    print()
    print()

# plot average costs
plt.errorbar(N, avg_costs, yerr=std_devs, fmt='-o', color='b', ecolor='r', capsize=5, label='Average Cost & SD')
plt.ylim(2000, 3000)
#
target = round(circuit_data[0]["Expected Cost"], 1)
plt.axhline(y=target, color='g', linestyle='--', linewidth=2, label=f'Expected Cost: {target}')
plt.title("Average Cost versus Number of Shots (using 20 repeats per experiment)")
plt.xlabel("Number of Shots")
plt.ylabel("Average QUBO Cost")
plt.grid(True, linestyle='--', alpha=0.6) # adds a clean background grid
plt.legend()

# display the chart
plt.show()

print()
print('''The average SE will typically decrease as N increases because SE = SD / sqrt(N), and SD 
generally stays consistent across runs. However, any individual run has randomly generated 
outcomes/measurements, so one run at a higher N won't necessarily have a lower SE; the average SE
over many runs at the higher N value will typically be lower.''')


# In[50]:


print("PART D - INTRODUCE NOISY SIMULATION\n")

# introduce 3 types of error (tuneable:
# Single qubit gate error = 0.1 % (error when applying a single gate, like rz or rx)
# Two-qubit gate error = 1% (error when applying entangling gates)
# Readout error = 2% (error when reporting the measured bit)

from qiskit_aer.noise import (
    NoiseModel, # container that holds the errors we define
    depolarizing_error, # models a gate operation that someimtes introduces random Pauli errors
    ReadoutError # models classical mistakes when converting a qubit's measurement into a 0 or 1
)

from qiskit import transpile

# get 2-qubit gate count. This is an important metric as 2-qubit gates are more error prone on real hardware
two_qubit_gate_count = sum(
    1 for instruction in measured_qc.data
    if instruction.operation.num_qubits == 2
)

print(f"1-Layer circuit's 2-qubit gate count = {two_qubit_gate_count}")
print()

# these parameters will control the strength of the depolarizing channel; they are not the exact
# probability of getting an incorrect output
p1 = 0.001 # single qubit gate error parameter
p2 = 0.01 # for two-qubit gates

error_1q = depolarizing_error(p1, 1) # creates a 1-qubit depolarizing channel
error_2q = depolarizing_error(p2, 2) # creates a 2-qubit depolarizing channel

# define readout error - row # (0 or 1) represents true outcome, column # represents reported outcome
# the elements in the array are the probabilities
readout_error = ReadoutError([
    [0.98, 0.02],
    [0.02, 0.98]
])

# create initially empty noise model
noise_model = NoiseModel()

# apply single-qubit gate error to single-qubit gates
noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'rz', 'h'])

# apply two-qubit gate error to two-qubit gates
noise_model.add_all_qubit_quantum_error(error_2q, ['rzz'])

# apply readout error to every qubit measurement
noise_model.add_all_qubit_readout_error(readout_error)

num_shots = 10000 # tuneable

# create noise simulator
noisy_simulator = AerSimulator(
    noise_model=noise_model,
    seed_simulator=42 # keep trials consistent to compare with ideal simulator
)

# calculate transpiled depth (transpiled circuit will be used in a later section, shown here for context)
# note: there shouldn't really be a difference in transpiled depth between noisy and ideal (in this case)
print("Original circuit depth:", measured_qc.depth())
transpiled = transpile(
    measured_qc,
    backend=noisy_simulator, # backend in this case is the simulator, but later on will be actual hardware
    optimization_level=3 # requests Qiskit's highest preset optimization level
)

print("Transpiled depth:", transpiled.depth())
print()

# run noisy simulator
print("Running simulator with noise parameters:")
print(f"Depolarizing error  = {p1 * 100}% on single-qubit gates (rx, rz), {p2 * 100}% on two-qubit gates (rzz)")
print(f"Measurement readout error = {100 * readout_error.probabilities[0][1]}%")
print()

noisy_counts = noisy_simulator.run(measured_qc, shots=num_shots).result().get_counts()
noisy_cost = sum([frequency * get_QUBOcost(K, h, J, [int(digit) for digit in bitstring][::-1]) 
                  for bitstring, frequency in noisy_counts.items()]) / num_shots
noisy_p_optimal = noisy_counts['0000000100'] / num_shots
noisy_p_feasible = sum(
    frequency for bitstring, frequency in noisy_counts.items()
    if int(bitstring[-1]) * 50 + int(bitstring[-2]) * 75 + int(bitstring[-3]) * 100 >= 100
) / num_shots

print(f"Noisy expected cost = {round(noisy_cost, 1)}")
print(f"Noisy probability of optimal state (A OFF, B OFF, C ON, all slack bits OFF) = {round(100 * noisy_p_optimal, 3)}%")
print(f"Noisy probability of feasible state (cooling >= 100 kW) = {round(100 * noisy_p_feasible, 1)}%") 

print()
print()
# create ideal (=non-noisy) simulator for comparison
ideal_simulator = AerSimulator(
    seed_simulator=42
)

# run ideal simulator
print("Running ideal simulator (no noise)...")

ideal_counts = ideal_simulator.run(measured_qc, shots=num_shots).result().get_counts()
ideal_cost = sum([frequency * get_QUBOcost(K, h, J, [int(digit) for digit in bitstring][::-1]) 
                  for bitstring, frequency in ideal_counts.items()]) / num_shots
ideal_p_optimal = ideal_counts['0000000100'] / num_shots
ideal_p_feasible = sum(
    frequency for bitstring, frequency in ideal_counts.items()
    if int(bitstring[-1]) * 50 + int(bitstring[-2]) * 75 + int(bitstring[-3]) * 100 >= 100
) / num_shots

print(f"Ideal expected cost = {round(ideal_cost, 1)}")
print(f"Ideal probability of optimal state (A OFF, B OFF, C ON, all slack bits OFF) = {round(100 * ideal_p_optimal, 3)}%")
print(f"Ideal probability of feasible state (cooling >= 100 kW) = {round(100 * ideal_p_feasible, 1)}%") 


print('''Deeper QAOA might improve ideal performance as additional layers will explore additional 
probability distributions, some of which may better favor low-cost states. However, real quantum
hardware has noise, and that noise increases as the circuit gains more layers and gates. The added
noise may actually degrade the result in reality.''')


# In[57]:


print("PART E - PREPARE IBM QUANTUM ACCESS\n")

from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    token="#####", # hidden for security purposes
    instance="#####", # same
    set_as_default=True,
    overwrite=True
)

service = QiskitRuntimeService()

backends = service.backends(
    operational=True,
    simulator=False
)

print("Available Processors:")
print()

for backend in backends:
    print(backend.name, backend.num_qubits)

backend = service.least_busy(
    operational=True,
    simulator=False,
    min_num_qubits=10
)

print()
print("Selected processor: ", backend.name)


# In[65]:


print("PART F - TRANSPILE\n")
print()

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager

qc = circuit_data[0]["Quantum Circuit"].copy() # same as measured_qc, just creating another copy to isolate them
qc.measure_all()

pm = generate_preset_pass_manager(
    backend=backend, # real quantum hardware this time
    optimization_level=3,
    translation_method="translator"
)

hardware_qc = pm.run(qc)

print("Original depth: ", qc.depth())
print("Transpiled depth: ", hardware_qc.depth())
print("Transpiled gate counts:")
for gate, count in hardware_qc.count_ops().items():
    print(f"    {gate}: {count}")

two_qubit_gate_count = sum(
    1 for instruction in hardware_qc.data
    if instruction.operation.num_qubits == 2
)

print("2-qubit gate count: ", two_qubit_gate_count)



# In[68]:


print("PART F (CONT.) - RUN ON REAL IBM HARDWARE\n")
print()

from qiskit_ibm_runtime import SamplerV2 as Sampler

sampler = Sampler(mode=backend) # backend is the IBM processor selected above

job = sampler.run([hardware_qc], shots=2000)

print("Job ID: ", job.job_id())
print("Job Status: ", job.status())



# In[78]:


print("PART G - DECODE HADWARE RESULTS AND COMPARE")
print()

result = job.result()
hardware_counts = result[0].data.meas.get_counts()

print("Total shots: ", sum(hardware_counts.values()))
print("Number of distinct outcomes: ", len(hardware_counts))

# calculate average cost
hardware_cost = 0
for bitstring, frequency in hardware_counts.items():
    assignment = [int(digit) for digit in bitstring][::-1]
    hardware_cost += frequency * get_QUBOcost(K, h, J, assignment)

avg_hardware_cost = hardware_cost / 2000

p_optimal_hardware = hardware_counts['0000000100'] / 2000
p_feasible_hardware =  sum(
    frequency for bitstring, frequency in hardware_counts.items()
    if int(bitstring[-1]) * 50 + int(bitstring[-2]) * 75 + int(bitstring[-3]) * 100 >= 100
) / 2000

print(f"Hardware's average QUBO cost: {round(avg_hardware_cost, 1)}")
print(f"Probability of optimal state (A OFF, B OFF, C ON, all slack bits OFF) using hardware: {round(100 * p_optimal_hardware, 5)}%")
print(f"Probability of measuring a feasible state (cooling >= 100 kW) using hardware: {round(100 * p_feasible_hardware, 1)}%")


# In[92]:


print("PART G (CONTD.) - TABULATION")
print()

import pandas as pd

row_labels = [  "Method",
                "Average Expected QUBO Cost",
                "Probability of Optimal Outcome (%)",
                "Probability of Feasible Outcome (%)",
                "Shot Count",
                "Circuit Depth"
                ]

methods =    [
              "Classical Optimum", 
              "Ideal Exact QAOA (1 layer)", 
              "Ideal Finite-Shot QAOA (1 layer)",
              "Noisy Simulator (1 layer)",
              "IBM Hardware (1 layer)"
             ]

expected_costs = [
                  25, 
                  round(circuit_data[0]["Expected Cost"], 1), 
                  round(ideal_cost, 1), 
                  round(noisy_cost, 1),
                  round(avg_hardware_cost, 1)
                 ]

prob_of_optimal = [
                   100, 
                   circuit_data[0]["Probability of Optimal State (A OFF B OFF C ON)"],
                   round(100 * ideal_p_optimal, 3),
                   round(100 * noisy_p_optimal, 3),
                   round(100 * p_optimal_hardware, 5)
                  ]

prob_of_feasible = [
                   100,
                   circuit_data[0]["Probability of a Feasible HVAC State (cooling >= 100)"],
                   round(100 * ideal_p_feasible, 1),
                   round(100 * noisy_p_feasible, 1),
                   round(100 * p_feasible_hardware, 1)
                   ]

shot_counts = [
              "N/A",
              "N/A",
              10000,
              10000,
              2000
              ]

circuit_depth = [
                "N/A",
                measured_qc.depth(),
                measured_qc.depth(),
                measured_qc.depth(),
                hardware_qc.depth()
                ] 

data = {
    "Method": methods,
    "Average Expected QUBO Cost": expected_costs,
    "Probability of Optimal Outcome (%)": prob_of_optimal,
    "Probability of Feasible Outcome (%)": prob_of_feasible,
    "Shot Count": shot_counts,
    "Circuit Depth": circuit_depth
}

# create the DataFrame
df = pd.DataFrame(data).style.set_properties(
    **{'text-align': 'center'}).set_table_styles(
    [{'selector': 'th', 'props': [('text-align', 'center')]}])

df


# In[ ]:





# In[ ]:




