
# In[130]:


# module 7

print("PART A - VARIABLES AND CONSTRAINTS\n")

print('''a,b,c are binary variables indicating whether each HVAC unit is on or off, via 1 or 0. 
P(a,b,c) = 10a + 17b + 25c
C(a,b,c) = 50a + 75b + 100c >= 100
The cooling inequality is not an unconstrained QUBO. the P function is the one to minimize, 
but the minimization is constrained by the cooling inequality.''')


# In[138]:


print("PART B - QUADRATIC WARM-UP\n")

import numpy as np
from itertools import product

def non_expanded(u, v): # u and v are binary variables
    return (3*u + 2*v - 4) ** 2 

def expanded(u, v):
    return -15*u - 12*v + 12*u*v + 16

bit = [0, 1] # options for each binary variable
test_cases = list(product(bit, repeat=2))
for case in test_cases:
    u = case[0]
    v = case[1]
    print(f"For u = {u} and v = {v}:")
    print(f"Non-expanded --> (3u + 2v - 4)^2 = (3 x {u} + 2 x {v} - 4)^2 = {non_expanded(u,v)}")
    print(f"Expanded --> -15u - 12v + 12uv + 16 = -15 x {u} - 12 x {v} + 12 x {u} x {v} + 16 = {expanded(u,v)}")
    if non_expanded(u,v) == expanded(u,v):
        print("Expansion valid!")
    else:
        print("Expansion invalid")
    print()


# In[159]:


print("PART C - SLACK ENCODING\n")

'''def slack(y0, y1, y2, y3, y4, y5, y6):
    s = y0 + 2*y1 + 4*y2 + 8*y3 + 16*y4 + 32*y5 + 64*y6
    return s'''

def slack(s):
    binary_s = format(s, 'b')
    return [int(digit) for digit in binary_s][::-1]  

test_nums = [0, 25, 125]
for num in test_nums:
    y = slack(num)
    padding_needed = 7 - len(y)
    y += [0] * padding_needed
    print(f"slack encoding of {num}:")
    for i in range(len(y)):
        print(f"y{i} = {y[i]}")
    print()

print(''' 7 bits works because the biggest possible cooling difference is when all 3 units are
on --> 50 + 75 + 100 = 225 kW, which is 125 kW above 100. adding 2^0 thru 2^6 gives:
1 + 2 + 4 + 8 + 16 + 32 + 64 = 127 which is greater than 125, so all possible cases are covered.''')
    


# In[179]:


print("PART D - CONSTRUCT Q\n")

# power coefficients
p = [10, 17, 25, 0, 0, 0, 0, 0, 0, 0]

# cooling-difference coefficients
d = [50, 75, 100, -1, -2, -4, -8, -16, -32, -64]

# penalty coefficient
lamda = 1 # can be tuned

# construct Q
Q = np.zeros((len(p), len(p)))
for i,j in np.ndindex(Q.shape):
    if i < j:
        Q[i][j] = 2 * lamda * d[i] * d[j]
    elif i == j:
        Q[i][j] = p[i] + lamda * (d[i] ** 2 - 200 * d[i])
    else:
        Q[i][j] = 0

np.set_printoptions(
    precision=2, # limit to 2 decimal places
    suppress=True, # suppress scientific notation
    linewidth=100 # keep line width consistent
)
print(f"Q Matrix = \n\n{Q}")
        
             
    
    


# In[194]:


print("PART E - ALGEBRAIC VERIFICATION\n")

import random

cooling_requirement = 100 # kW

bit = [0,1]
num_tests = 5
test_cases = []

# define binary vector z = a,b,c,y0,...y6) using randomly assigned bits
for i in range(num_tests):
    z = np.array(random.choices(bit, k=len(p)))
    QUBO_direct = np.dot(p,z) + lamda * (np.dot(d,z) - cooling_requirement) ** 2
    QUBO_simplified = lamda * cooling_requirement ** 2 + z.transpose() @ Q @ z
    print(f"For 10-bit assignment z = {z}:")
    print(f"Direct QUBO F(z) = 10a + 17b + 25c + λ(50a + 75b + 100c - 100 - s(y))^2 = {QUBO_direct}")
    print(f"Simplified QUBO F(z) = 10000λ + zᵀQz = {QUBO_simplified}")
    print()




# In[203]:


print("PART F - EXHAUSTIVE OPTIMIZATION\n")

import itertools

bit = [0, 1]
assignments = list(itertools.product(bit, repeat=len(p)))
QUBO_costs = [np.array(z).transpose() @ Q @ z for z in assignments] # ignore the constant term since it affects all assignments equally
optimal = assignments[np.argmin(QUBO_costs)]
lowest_cost = min(QUBO_costs)

optimal_cooling = optimal[0] * 50 + optimal[1] * 75 + optimal[2] * 100
optimal_consumption = optimal[0] * 10 + optimal[1] * 17 + optimal[2] * 25

def status(binary_var):
    if binary_var == 0:
        return "OFF"
    elif binary_var == 1:
        return "ON"
    else: 
        return None

print(f"Optimal state z = {optimal}")
print()
print(f"Cooling = {optimal_cooling} kW \nConsumption = {optimal_consumption} kW")
print()
print(f"QUBO Cost = {lowest_cost}")
print()
print(f"HVAC bits: a = Unit A = {optimal[0]} = {status(optimal[0])}, \
b = Unit B = {optimal[1]} {status(optimal[1])}, \
c = Unit C = {optimal[2]} = {status(optimal[2])}")
print()
print(f"Slack bits:  y0 = {optimal[3]}, y1 = {optimal[4]}, y2 = {optimal[5]}, \
y3 = {optimal[6]}, y4 = {optimal[7]}, y5 = {optimal[8]}, y6 = {optimal[9]}")
    


# In[233]:


# write it all as a function

cooling_requirement = 100 # kW
bit = [0, 1]

# power coefficients
p = [10, 17, 25, 0, 0, 0, 0, 0, 0, 0]

# cooling-difference coefficients
d = [50, 75, 100, -1, -2, -4, -8, -16, -32, -64]

true_optimal = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # A off, B off, C on, zero residual
print(f"True Optimal = {true_optimal}")
print()
print()

total_assignments = list(itertools.product(bit, repeat=len(p)))

def status(binary_var):
        if binary_var == 0:
            return "OFF"
        elif binary_var == 1:
            return "ON"
        else: 
            return None

def optimal(lamda, assignments):
    Q = np.zeros((len(p), len(p)))
    for i,j in np.ndindex(Q.shape):
        if i < j:
            Q[i][j] = 2 * lamda * d[i] * d[j]
        elif i == j:
            Q[i][j] = p[i] + lamda * (d[i] ** 2 - 200 * d[i])
        else:
            Q[i][j] = 0

    QUBO_costs = [lamda * cooling_requirement ** 2 + np.array(z).transpose() @ Q @ z for z in assignments] 
    optimal = assignments[np.argmin(QUBO_costs)]
    lowest_cost = min(QUBO_costs)
    
    optimal_cooling = optimal[0] * 50 + optimal[1] * 75 + optimal[2] * 100
    optimal_consumption = optimal[0] * 10 + optimal[1] * 17 + optimal[2] * 25

    print(f"For λ = {lamda}:")
    print()
    print(f"Optimal state z = {optimal}")
    print()
    print(f"Cooling = {optimal_cooling} kW \nConsumption = {optimal_consumption} kW")
    print()
    print(f"Residual = cooling - requirement = {optimal_cooling} - {cooling_requirement} = \
{optimal_cooling - cooling_requirement} kW")
    print()
    print(f"QUBO Cost = {lowest_cost}")
    print()
    print(f"HVAC bits: a = Unit A = {optimal[0]} = {status(optimal[0])}, \
    b = Unit B = {optimal[1]} {status(optimal[1])}, \
    c = Unit C = {optimal[2]} = {status(optimal[2])}")
    print()
    print(f"Slack bits:  y0 = {optimal[3]}, y1 = {optimal[4]}, y2 = {optimal[5]}, \
    y3 = {optimal[6]}, y4 = {optimal[7]}, y5 = {optimal[8]}, y6 = {optimal[9]}")
    print()    
    return optimal

test_lamdas = [1, 0.1, 0.01]
for lamda in test_lamdas:
    test_optimal = optimal(lamda=lamda, assignments=total_assignments)
    if np.array_equal(true_optimal, test_optimal):
        print(f"Correct solution. Penalty λ = {lamda} is sufficiently large.")
    else:
        print(f"Incorrect solution. Penalty λ = {lamda} is insufficiently large.")
    print()
    print()
    


# In[243]:


print("PART H - COMPARE FORMULATIONS\n")

x = [[50, 75, 100],
     [10, 17, 25]]

bit = [0,1]

slack_assignments = list(itertools.product(bit, repeat=len(p)-3)) # only iterate thru slack bits

λ = 1

def optimize(configuration): # configuration is in the form (0,0,0) for (a,b,c)
    total_assignments = [configuration + slack_assignment for slack_assignment in slack_assignments]
    optimal(lamda=1, assignments=total_assignments)

configurations = list(itertools.product(bit, repeat=3)) # these are HVAC unit configurations

# from mod. 6:
cooling = [(x @ np.array(configuration))[0] for configuration in configurations]
consumption = [(x @ np.array(configuration))[1] for configuration in configurations]
cost = [consumption[i] + λ * max(0, 100 - cooling[i]) ** 2 for i in range(len(consumption))]

for i in range(len(configurations)):
    a,b,c = configurations[i][0], configurations[i][1], configurations[i][2]
    print(f"State: A = {status(a)}, B = {status(b)}, C = {status(c)}")
    optimize(configurations[i])
    print(f"Hamiltonian hinge-squared cost = {cost[i]}")
    print()
    print()

print('''For each row, the hamiltonian hinge_squared cost equals the minimum QUBO cost. 
This verifies that the two formulations are equivalent after optimizing the slack bits.
The minimum costs differ between rows, because each HVAC configuration has different
power consumption and cooling capacity.''')


# In[245]:


print("PART I - WRITTEN REFLECTION\n")

print('''slack bits allow us to encode the residual inside the optimization function,
thus making the function unconstrained. 
the final objective is quadratic because binary variables equal their square:
0 = 0^2 and 1 = 1^2. So something like (a1 + a2 + a3)^2 will still be quadratic after expansion.
The diagonal Q entries are the effects of each binary variable on its own, and the off-diagonals
are the interactions between binary variables.
This encoding has scalability limitations because the number of possible binary assignments 
grows exponentially with the number of variables. Although the QUBO matrix grows quadratically,
exhaustive optimization becomes increasingly expensive. Additional slack variables also increase 
the number of binary variables and interactions.''')



# In[ ]:




