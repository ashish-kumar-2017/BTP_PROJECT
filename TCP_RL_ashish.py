import random
import itertools
# Define state and action space size
state_size = 3
min_value = 0
min_change = -10
count_action = 100
max_change = +10
max_value = 100
num_episodes = 1000
gamma = 0.9  # Discount factor
epsilon = 1.0  # Initial exploration probability
epsilon_min = 0.01  # Minimum epsilon value
epsilon_decay = 0.98  # Decay rate for epsilon it should be high since multipy

# Define the numbers and vector size
numbers = range(-10, 11)  # Numbers from 0 to 100 inclusive
vector_size = 3

# Generate all permutations with repetition
permutations = list(itertools.product(numbers, repeat=vector_size))

# Q-value storage and Returns list
Q = {}  # Q(s, a)
Returns = {}  # Returns for state-action pairs
policy = {}  # Policy: π(s) -> a


# Function to choose action based on the policy or exploration
def choose_action(state, epsilon=0.1):
    # Exploration vs exploitation: Choose random action with probability epsilon
    # random.random() -generate a floating number between [0,1.0)
    if random.random() < epsilon:
        # The random.random() < epsilon condition ensures that the agent doesn't always exploit its current knowledge,
        # which helps avoid local optima and promotes better learning.
        action = [random.randint(min_change, max_change) for _ in range(state_size)]
    else:
        # Exploitation: Choose the action that maximizes Q-value for the current state
        #directly choosing from policy since it include best action for current action
        if tuple(state) in policy:
            action = policy[tuple(state)]
        else:
            # Default to random if no policy
            action = [random.randint(min_change, max_change) for _ in range(state_size)]
    return action


# Function to apply the action and get the next state
def apply_action(action, state):
    # Step 1: Compute the tentative next state
    # I am providing first next state by sum of present state and action and improve it additionally
    next_state = [state[i] + action[i] for i in range(len(action))]

#if after adding next state goes less than 0 then make it 0
#if after adding next state goes greater than 100 make it 100
    #to bring it within range
    next_state = [max(0, min(rate, 100)) for rate in next_state]

#since below code has been included in generate episode function
#since real network simulation will not take such a next state that exceed link capacity
    # Check and adjust for link capacity
    # n0, n1, n2 = next_state
    # total_rate = sum(next_state)
    #
    # if (n0 + n1 < 100) and (n0 + n1 + n2 < 128):
    #     print("Final condition satisfied: Success!")
    #
    # # Condition 2: If the sum of the first two state vectors is greater than 100
    # if n0 + n1 > 100:
    #     scaling_factor2 = 100 // (n0 + n1)
    #     # Apply scaling to n0 and n1
    #     next_state[0] = n0 * scaling_factor2
    #     next_state[1] = n1 * scaling_factor2
    #
    #
    # # Condition 1: If the sum of all state vectors is greater than 128
    # if n0 + n1 + n2 > 128:
    #     scaling_factor1 = 128 // total_rate
    #     next_state[2] = n2 * (2*scaling_factor1)
    #     # next_state = [rate * scaling_factor1 for rate in next_state]
    #
    # # Enforce minimum and maximum data rate bounds
    # #after scaling it may lie outside range so
    # # each next state data rate should lie within 0 and 100
    # next_state = [max(min_value, min(rate, max_value)) for rate in next_state]

    return next_state


# Function to calculate the reward for the given state
def calculate_reward(state):
    # Unpack state vector into s0, s1, s2
    s0, s1, s2 = state

    reward = 0

#link capacity related reward
    # Link capacity thresholds
    link1_capacity= 100
    link2_capacity= 128

    # Calculate sums for conditions
    sum_link1 = s0 + s1
    sum_link2 = s0 + s1 + s2

    # Condition 1: Check link 1 capacity
    if sum_link1 < link1_capacity:
        # Scaled reward for staying under threshold and closer to link capacity
        reward += 100 * (1 - sum_link1 / link1_capacity)
    else:
        reward -= 100  # Scaled penalty for exceeding threshold

    # Condition 2: Check link 2 capacity
    if sum_link2 < link2_capacity:
        reward += 100 * (1 - sum_link2 / link2_capacity)  # Scaled reward
    else:
        reward -= 100  # penalty

    # Combined condition: Both conditions met
    # Bonus for satisfying both conditions
    if sum_link1 < link1_capacity and sum_link2 < link2_capacity:
        reward += 100


# maximise utility related reward
    scaling_factor_link1 = (100 - (s0 + s1))
    scaling_factor_link2 = (128 - (s0 + s1 + s2))
    if scaling_factor_link1 > 0:
        reward += 1000 // scaling_factor_link1
    if scaling_factor_link1 < 0:
        reward -= 1000 // abs(scaling_factor_link1)


    if scaling_factor_link2 > 0:
        reward += 1000 // scaling_factor_link2
    if scaling_factor_link2 < 0:
        reward -= 1000 // abs(scaling_factor_link2)



#delay bound related reward

    if s0 == 100 or s0 == 128 or s1 == 100 or s1 == 128 or s2 == 100 or s2 == 128:
        return reward  # Skip termination check if division by zero would occur

    # Condition 1: (1/(128-s2)) < 0.0002//(0.01 0.03 0.04)
    condition1 = 0<(1 / (128 - s2)) < 0.01

    # Condition 2: (1/(100-s0)) + (1/(128-s0)) < 0.0002
    condition2 = 0<((1 / (100 - s0)) + (1 / (128 - s0))) < 0.02

    # Condition 3: (1/(100-s1)) + (1/(128-s1)) < 0.0002
    condition3 = 0<((1 / (100 - s1)) + (1 / (128 - s1))) < 0.03



    # Check if all condition is met most reward
    if condition1 and condition2 and condition3:
        reward += 750
    # Check if any of two condition is met somewhat more reward
    elif condition1 and condition2 or condition1 and condition3 or condition2 and condition3:
        reward += 250
    # Check if anyone condition is met less reward
    elif condition1 or condition2 or condition3:
        reward += 170
    else:
        reward-=750


    return reward


def check_termination(state):
    # Unpack state vector into s0, s1, s2
    s0, s1, s2 = state

    # Check for potential division by zero (avoid if s0, s1, or s2 is 100 or 128)
    if s0 == 100 or s0 == 128 or s1 == 100 or s1 == 128 or s2 == 100 or s2 == 128:
        return False  # Skip termination check if division by zero would occur

    # Condition 1: (1/(128-s2)) < 0.01
    condition1 = 0<(1 / (128 - s2)) < 0.01

    # Condition 2: (1/(100-s0)) + (1/(128-s0)) < 0.01
    condition2 = 0<((1 / (100 - s0)) + (1 / (128 - s0))) < 0.02

    # Condition 3: (1/(100-s1)) + (1/(128-s1)) < 0.01
    condition3 = 0<((1 / (100 - s1)) + (1 / (128 - s1))) < 0.03

    # Check if all conditions are met
    if condition1 and condition2 and condition3:
        return True  # Terminate
    else:
        return False  # Continue


def generate_episode(epsilon):
    state = [50, 50, 128]

    # Store the episode history
    episode_history = []
    temp_count_action = 0
    total_reward = 0.0
    G = 0.0

    print(f"Initial State: {state}")  # Debugging: Print initial state

    while True:

        action = choose_action(state, epsilon)
        # inccrease count of action taken //optional too observe total action taken in one episode
        temp_count_action = temp_count_action + 1

        # Calculate reward and append state-action-reward tuple
        reward = calculate_reward(state)
        total_reward += reward
        episode_history.append((state, action, reward))

        # Apply action to get next state
        next_state = apply_action(action, state)
        n0, n1, n2 = next_state

        if (n0 + n1 < 100) and (n0 + n1 + n2 < 128):
            print("Final condition not satisfied: Unsuccessful!")
            break
        # Check termination condition
        if check_termination(state):
            print(f"Stability will occur at rates of Host0,Host1 and Host2: {state},With total action taken: {temp_count_action}.")  # Debugging: Print when termination is met
            print(f"Total Accumulated Reward in one episode: {total_reward}")
            break

        state = next_state


    # Backward update (update Q-values and policy)
    temp_count_action = 0
    G = 0
    for state, action, reward in reversed(episode_history):
        sa = (tuple(state), tuple(action))

        G = gamma * G + reward  # Discounted future reward
        if sa not in Returns:
            Returns[sa] = G
        else:
            Returns[sa] += G  # Accumulate return for repeated visits

        # Update the Q-value (average of the returns)
        if sa not in Q:
            Q[sa] = [Returns[sa]]
        else:
            Q[sa].append(Returns[sa] / len(Q[sa]))  # Average return

 # Update the policy (choose the action that maximizes Q(s, a))
        # Initializes the maxQ  variable to a very low   value(-infinity),
        maxQ = -float('inf')
        best_action = None
 # This loop  iterates  over  all possible actions for the current state.
        for vector in permutations:
            action_option = vector
            sa_option = (tuple(state), tuple(action_option))
            if sa_option not in Q or len(Q[sa_option]) == 0:
                continue
            #the q value below represents state action pair return
            q_value = max(Q[sa_option])
            if q_value > maxQ:
 # If the state - action pair exists in Q, the  code retrieves the maximum  Q - value from all episode
                maxQ = q_value
                best_action = action_option
        policy[tuple(state)] = best_action

 # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    return epsilon




#
# def evaluate_policy():
#     """Function to evaluate the learned policy."""
#     total_return = 0
#     num_evaluations = 0
#
#     for state, action in policy.items():
#         # Get the Q-value for the state-action pair
#         sa = (tuple(state), tuple(action))
#         if sa in Q:
#             # Get the average Q-value for this state-action pair
#             q_value = max(Q[sa])
#             total_return += q_value
#             num_evaluations += 1
#
#     # Compute the average return for all states
#     if num_evaluations > 0:
#         average_return = total_return / num_evaluations
#         print(f"Average Return for Learned Policy: {average_return}")
#     else:
#         print("No evaluations available.")
#
#
#



# Function to run the Monte Carlo control process
def monte_carlo_control():
    epsilon = 1.0  # Initial epsilon
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}:")
        epsilon = generate_episode(epsilon)

        # # Display policy for each state
        # print("Updated Policy:")
        # for state, action in policy.items():
        #     print(f"State: {state} | Action: {action}")
        # After running the episodes, evaluate the learned policy
    print("Evaluating the learned policy:")
    evaluate_policy()


# Run the Monte Carlo Control process
if __name__ == "__main__":
    random.seed()  # Seed the random number generator
    monte_carlo_control()  # Run the Monte Carlo Control update
