import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    transitions = np.asarray(transitions)
    rewards = np.asarray(rewards)
    new_values = []
    for s in range(transitions.shape[0]):

        total = []

        for a in range(transitions.shape[1]):
            reward = rewards[s,a]
            action = 0
            for s_prime in range(transitions.shape[2]):
                action+= gamma * transitions[s,a,s_prime]*values[s_prime]

            total.append(reward + action)
        
        new_values.append(max(total))

    return new_values
    
    