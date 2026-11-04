
# module 8

print("PART A - BINARY TO SPIN MAPPING\n")

import numpy as np
from itertools import product, combinations

def status(bit):
    if bit == 0:
        return "OFF"
    elif bit == 1:
        return "ON"
    else:
        return None

def spin(z): # z is a binary variable - either 0 or 1
    return 1 - 2 * z # z = 0 maps to 1, z = 1 maps to -1

bit = [0, 1]
configurations = list(product(bit, repeat=3))
for configuration in configurations:
    spin_form = tuple([spin(unit) for unit in configuration])
    print(f"Configuration: {configuration} --> Spin-variable form: {spin_form}")
    for i in range(len(spin_form)):
        if configuration[i] == 0 and spin_form[i] == 1:
            continue
        elif configuration[i] == 1 and spin_form[i] == -1:
            continue
        else:
            print(f"Error - spin for element at index {i} not mapped correctly")
    print()


# In[263]:


print("PART B - ONE-QUBIT PAULI Z-CHECK\n")

Z = np.array([
    [1, 0],
    [0, -1]
])

I = np.array([
    [1, 0],
    [0, 1]
])

n  = (I - Z) / 2

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

bit = [0,1]
num_qubits = 1
num_basis_states = len(bit) ** num_qubits
basis_vectors = []
for i in range(num_basis_states):
    basis_vector = np.zeros(num_basis_states)
    basis_vector[i] = 1
    basis_vectors.append(basis_vector)
    print(f"Applying Pauli operator n = (I - Z) / 2 to |{i}> = {basis_vector} --> result = {n @ basis_vector}")
    eigenvalue = get_eigenvalue(n, basis_vector)
    print(f"eigenvalue = {eigenvalue[1]}")
    print()

print('''n represents an ON/OFF bit because the resulting eigenvalue will indicate whether the
unit is ON or OFF. If the original state is |0> = OFF, the resulting eigenvalue is 0, indicating
OFF. and vice versa.''')



# In[350]:


print("PART C - TWO-VARIABLE ALGEBRA WARM-UP\n")

from itertools import combinations

print(''' f(u,v) = 3u + 2v + 4uv + 5
 u = (1 - s1) / 2; v = (1 - s2)/2
 f(s1, s2) = 3/2 - (3/2)s1 + 1 - s2 + 1 - s1 - s2 + s1s2 + 5
 f(s1,s2) = (-5/2)s1 + (-2)s2 + (1)s1s2 + (17/2)
 c = 17/2
 alpha1 = -5/2; alpha2 = -2
 beta12 = 1 ''')

print()
print()

num_vars = 2

def test_QUBO(u, v): # u and are binary variables --> 0 or 1
    f = 3 * u + 2 * v + 4 * u * v + 5
    return f

# ising coefficients
c = 17 / 2
alpha = [-5/2, -2]
beta = [[0, 1],
        [0, 0]]

def test_ising(s1, s2): # s1 and s2 are Pauli variables --> -1 or +1
    # get combinations of indices - e.g. (0,1) = (s1,s2)
    s = [s1, s2]
    s_pairs = list(combinations(range(len(s)), 2))
    f = c + sum([alpha[i] * s[i] for i in range(len(s))]) + \
    sum([beta[i][j] * s[i] * s[j] for i,j in s_pairs])
    return f

binary_bit = [0, 1]
pauli_bit = [1, -1]

binary_combinations = list(product(binary_bit, repeat=num_vars))
pauli_combinations = list(product(pauli_bit, repeat=num_vars))

for i in range(len(binary_combinations)):
    u = binary_combinations[i][0]
    v = binary_combinations[i][1]
    s1 = pauli_combinations[i][0]
    s2 = pauli_combinations[i][1]
    QUBO_result = test_QUBO(u, v)
    ising_result = test_ising(s1, s2)
    print(f"For u = {u} and v = {v}:")
    print(f"QUBO f(u,v) = 3u + 2v + 4uv + 5 = {QUBO_result}")
    print()
    print(f"Substituting Pauli variables u = (1 - s1)/2 , v = (1 - s2) / 2...")
    print(f"--> Ising coefficients: \nc = {c}, \nalpha = {alpha}, \nbeta = {beta}")
    print(f"f(s1,s2) = c + sum(alpha_i * s_i) + sum(beta_ij * s_i * s_j) = {ising_result}")
    print()
    print()


# In[351]:


print("PART D - EXTRACT THE HVAC QUBO COEFFICIENTS\n")

import random

# from module 7, slightly modified:

cooling_requirement = 100 # kW
bit = [0, 1]

# power coefficients
power = [10, 17, 25, 0, 0, 0, 0, 0, 0, 0]

# cooling-difference coefficients
diff = [50, 75, 100, -1, -2, -4, -8, -16, -32, -64]

total_assignments = list(product(bit, repeat=len(power)))

def get_QUBOcoefficients(lamda, p, d, constraint):
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
    return Q, K, h, J

def get_QUBOcost(K, h, J, z):
    if len(h) == len(z) == len(J):
        pass
    else:
        print("Error: Invalid QUBO inputs, dimension mismatch.")
    z_pairs = list(combinations(range(len(z)), 2))    
    F = K + sum([h[i] * z[i] for i in range(len(h))]) + sum([J[i][j] * z[i] * z[j] for i,j in z_pairs])
    return F

def get_direct_cost(p, d, z, lamda, constraint): # avoids Q matrix entirely
    F = np.dot(p,z) + lamda * (np.dot(d,z) - constraint) ** 2
    return F

test_lamda = 1
Q, K, h, J = get_QUBOcoefficients(lamda=test_lamda, p=power, d=diff, constraint=cooling_requirement)

print(f"For λ = {test_lamda}:")
print(f"Q = \n{Q} \nK = {K} \nh = {h} \nJ = \n{J}")
print()
print()
test_assignments = random.sample(total_assignments, k=5)
for z in test_assignments:
    QUBOcost = get_QUBOcost(K=K, h=h, J=J, z=z)
    direct_cost = get_direct_cost(p=power, d=diff, z=z, lamda=1, constraint=cooling_requirement)
    print(f"for assignment z = {z}: \nQUBO cost = {QUBOcost} \nDirect cost = {direct_cost}")
    print()


# In[352]:


print("PART E - CONSTRUCT ISING COEFFICIENTS\n")

def get_IsingCoefficients(K, h, J):
    alpha = [-h[i]/2 - (1/4) * sum([J[min(i,j)][max(i,j)] for j in range(len(h)) if j != i]) \
             for i in range(len(h))]
    beta  = J / 4
    c = K + (1/2) * sum(h) + (1/4) * J.sum()
    return alpha, beta, c

alpha, beta, c = get_IsingCoefficients(K, h, J)
print(f"alpha coefficients = {alpha}")
print()
print(f"beta coefficients = \n{beta}")
print()
print(f"c coefficient = {c}")
print()
print('''The quadratic Jij term, when expanded using the Pauli substitutions z = .5(1-s),
produces 4 terms - one constant, one for s_i individually, one for s_j individually,
and one for the pairwise term s_i x s_j. the individual s_i and s_j terms contribute to the 
linear spin coefficients for each of those variables, respectively.''')


# In[353]:


print("PART F - VERIFY ALL 1,024 ASSIGNMENTS\n")

def get_IsingEnergy(alpha, beta, c, z): # z is the assignment
    if len(alpha) == len(beta) == len(z):
        pass
    else:
        return "Invalid Ising input - dimension mismatch"
    s = [spin(z[i]) for i in range(len(z))]
    pairs = list(combinations(range(len(s)),2))
    F = c + sum([alpha[i] * s[i] for i in range(len(s))]) + sum([beta[i][j] * s[i] * s[j] \
                                                                 for i,j in pairs])
    return F
    
max_discrepancy = 0
z_of_max_discrepancy = None
min_energy = float('inf')
optimal = None
flag = False # will be used to indicate any mismatches between cost formulas
print("Running QUBO, direct cost, and Ising Energy on 1,024 assignments...")
print()

for z in total_assignments:
    QUBOcost = get_QUBOcost(K=K, h=h, J=J, z=z)
    direct_cost = get_direct_cost(p=power, d=diff, z=z, lamda=1, constraint=cooling_requirement)
    IsingEnergy = get_IsingEnergy(alpha=alpha, beta=beta, c=c, z=z)
    if IsingEnergy < min_energy:
        min_energy = IsingEnergy
        optimal = z
    max_discrepancy_i = max(abs(QUBOcost - direct_cost), abs(QUBOcost - IsingEnergy), abs(IsingEnergy - direct_cost))
    if max_discrepancy_i > max_discrepancy:
        max_discrepancy = max_discrepancy_i
        z_of_max_discrepancy = z
    if np.allclose([QUBOcost, direct_cost, IsingEnergy], [QUBOcost] * 3):
        pass
    else:
        print(f"Error at assignment z = {z} -> discrepancy of {max_discrepancy_i}:")
        print(f"QUBO cost = {QUBOcost} \nDirect cost = {direct_cost} \nIsing Energy = {IsingEnergy}")
        print()
        flag = True

if flag == False:
    print("No significant discrepancies")
    if max_discrepancy != 0:
        print(f"max discrepancy = {max_discrepancy} at z = {z_of_max_discrepancy}")
    else:
        print("Max discrepancy = 0")

print()
unit_a = status(optimal[0])
unit_b = status(optimal[1])
unit_c = status(optimal[2])
print(f"Optimal state = {optimal} = A {unit_a}, B {unit_b}, C {unit_c}")
print(f"Minimum energy = Energy at optimal state = {min_energy}")


# In[354]:


print("PART G - BUILD SELECTED OPERATOR MATRICES\n")

from itertools import product

z = np.array([
    [1,0],
    [0,-1]
])

i = np.array([
    [1,0],
    [0,1]
])

# for 2 qubit system from part C
c_test = 17 / 2
alpha_test = [-5/2, -2]
beta_test = [[0, 1],
        [0, 0]]
Z0  = np.kron(z, i)
Z1  = np.kron(i, z)
# Z01 = np.kron(z, z)
Z = [Z0, Z1]
I = np.kron(i, i)
pairs = list(combinations(range(len(Z)), 2))
# build Ising Hamiltonian
H_ising = c_test * I + sum([alpha_test[j] * Z[j] for j in range(len(alpha_test))]) + sum([beta_test[j][k] * (Z[j] @ Z[k]) for j,k in pairs])
print(f"Ising Hamiltonian = H_c = \n{H_ising}")
print()

bit = [0,1]
num_qubits = 2
configurations = list(product(bit, repeat=num_qubits))

# basis_vectors = []
for i in range(len(configurations)):
    basis_vector = np.zeros(len(configurations))
    basis_vector[i] = 1
    # basis_vectors.append(basis_vector)
    ising_transform = H_ising @ basis_vector
    eigenvalue = get_eigenvalue(H_ising, basis_vector)
    ising_energy = get_IsingEnergy(alpha=alpha_test, beta=beta_test, c=c_test, z=np.array(configurations[i]))
    print(f"Ising transform on {basis_vector} = H_c x {basis_vector} = {ising_transform}")
    print(f"Eigenvalue = {eigenvalue[1]}")
    print(f"Classical Ising cost = {ising_energy}")
    print()
    


# In[367]:


print("PART H - GROUND STATE AND CONSTANT SHIFT\n")

def compare_IsingEnergies(alpha, beta, c, z): # z is the assignment
    if len(alpha) == len(beta) == len(z):
        pass
    else:
        return "Invalid Ising input - dimension mismatch"
    s = [spin(z[i]) for i in range(len(z))]
    pairs = list(combinations(range(len(s)),2))
    F_with_c = c + sum([alpha[i] * s[i] for i in range(len(s))]) + sum([beta[i][j] * s[i] * s[j] \
                                                                 for i,j in pairs])
    F_without_c = F_with_c - c
    return F_with_c, F_without_c

min_with_c = float('inf')
optimal_with_c = None

min_without_c = float('inf')
optimal_without_c = None

for assignment in total_assignments:
    F_with, F_without = compare_IsingEnergies(alpha=alpha, beta=beta, c=c, z=assignment)
    if F_with < min_with_c:
        min_with_c = F_with
        optimal_with_c = assignment
    if F_without < min_without_c:
        min_without_c = F_without
        optimal_without_c = assignment
print(f"Optimal state using Ising WITH constant term c --> {optimal_with_c}")
print(f"Optimal state using Ising WITHOUT constant term c --> {optimal_without_c}")
print()
print(f"Minimum Ising energy WITH constant term c --> {min_with_c}")
print(f"Minimum Ising energy WITHOUT constant term c --> {min_without_c}")
print()
print('''Without the constant c, the optimal assignment does not change as removing c affects all
assignments equally. however, the absolute cost of the optimal changes since c has been 
subtracted.''')
print()
if all(var == 0 for var in optimal_with_c[3:]) and all(var == 0 for var in optimal_without_c[3:]):
    print('''The optimal 10-bit assignment (for either method) has all slack bits equal to 0, 
which is correct as there is no residual for the configuration A off, B off, C on''')
else:
    print('''The optimal 10-bit assignment (for eitherone or both methods) does NOT have
    all slack bits equal to 0, which is incorrect as there is no residual for 
    the configuration A off, B off, C on''')


# In[368]:


print("PART H - WRITTEN REFLECTION\n")

print()
print('''binary-to-spin mapping is reversible via z = (1-s)/2.
A QUBO interaction becomes a ZZ term because the the Pauli substitution of z_i * z_J with s_i and
s_j produces a pairwise interaction s_i * s_j; when that gets replaced with Pauli Z matrices
the Zs for the corresponding s-bits get tensored together, creating ZZ (as opposed to ZI or 
IZ for individual bits).
The ising cost Hamiltonian is diagoanal because it is generated through Pauli-Z and 
identity operators, which are diagonal. The product and tensor product of diagonal matrices
is diagonal. Consequently, the computational basis states are eigenstates who eigenvalues are the
corresponding diagonal entry in Hc; that eigenvalue represents the cost of that basis state.
Each qubit represents something different (an HVAC unit, a slack bit, etc.), so misordering the 
bits can lead to applying an operator or coefficient to the wrong bit, which will impact the
result.
The ising hamiltonian provides an equation that can be optimized to find the best solution,
but we still need to manually search through the results to find that solution.
QAOA will do the searching for us and increase the probability of measuring low-cost
configurations, but it will require preparing a qubit into an initial state
that gives an amplitude for each of the 1,024 possible assignments. the ising hamiltonian also 
needs to be converted into a parametrized, unitary quantum operation that can be applied to 
a quantum state. A mixer (to mix amplitudes and configurations) will also need to be introduced.
Finally, we will alternate cost and mixer operations to find the minimal cost.''')


# In[ ]:




