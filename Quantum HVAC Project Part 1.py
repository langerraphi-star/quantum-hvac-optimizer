import numpy as np
import random

# unit A: 50 kW cooling capacity, 10 kW power consumption
# unit B: 75 kW cooling capacity, 17 kW power consumption
# unit C: 100 kW cooling capacity, 25 kW power consumption

cooling_demand = 100 # 100 kW needed for the building

states = [0, 1] # 0 = off, 1 = on

units = ["A", "B", "C"]

num_options = (len(states)) ** (len(units))

previous_states = []
success_states = {}

i = 1 # permutations counter
while len(previous_states) < num_options:
    
    A = random.choice(states)
    B = random.choice(states)
    C = random.choice(states)
    
    current_state = np.array([
        [A],
        [B],
        [C]
    ])

    current_state_string = ["off" if state[0] == 0 else "on" for state in current_state]

    if any(np.array_equal(current_state, state) for state in previous_states):
        continue

    previous_states.append(current_state)
    print()
    print()
    print(f"Option # {i}:")
    print(f"{units[0]} = {current_state_string[0]}")
    print(f"{units[1]} = {current_state_string[1]}")
    print(f"{units[2]} = {current_state_string[2]}")
    
    x = np.array([
        [50, 75, 100],
        [10, 17, 25]
    ])
    
    power = np.matmul(x, current_state)
    cooling = power[0][0]
    consumption = power[1][0]
    check = cooling >= cooling_demand
    print(f"Power Consumption = {consumption} kW")
    print(f"Cooling Output = {cooling} kW")
    print(f"Cooling Demand = {cooling_demand} kW")
    print(f"Demand met? {check}")

    if check:
        success_states[tuple(current_state_string)] = (cooling, consumption)

    i += 1

best_state = None
lowest_consumption = float('inf')
for state, result in success_states.items():
    if result[1] <= lowest_consumption:
        lowest_consumption = result[1]
        best_state = state
print()
print(f"Best State: {units[0]} = {best_state[0]}, {units[1]} = {best_state[1]}, {units[2]} = {best_state[2]}")
print(f"Cooling Output: {success_states[best_state][0]} kW")
print(f"Power Consumption: {lowest_consumption} kW")
    
    
    
    
    


