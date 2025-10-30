import numpy as np
import matplotlib.pyplot as plt
import random

class SARSAAgent:
    def __init__(self, env, eta=0.5, gamma=1.0, epsilon=0.1):
        self.env = env
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = np.zeros((env.height, env.width, 4))
        self.rewards_per_episode = []

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.get_actions())
        else:
            return np.argmax(self.q_values[state[0], state[1]])
    #############################################
    # IMPLEMENT ALGORITHM IN THE TRAIN FUNCTION #
    #############################################
    def train(self, num_episodes):
        pass
        return self.q_values, self.rewards_per_episode

class QLearningAgent:
    def __init__(self, env, eta=0.5, gamma=1.0, epsilon=0.1):
        self.env = env
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = np.zeros((env.height, env.width, 4))
        self.rewards_per_episode = []

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.get_actions())
        else:
            return np.argmax(self.q_values[state[0], state[1]])

    #############################################
    # IMPLEMENT ALGORITHM IN THE TRAIN FUNCTION #
    #############################################
    def train(self, num_episodes):
        pass
        return self.q_values, self.rewards_per_episode

class TDLambdaAgent:
    def __init__(self, env, eta=0.5, gamma=1.0, epsilon=0.1, lam=0.9):
        self.env = env
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.lam = lam
        self.v_values = np.zeros((env.height, env.width))
        self.rewards_per_episode = []

    def choose_action(self, state):
        actions = self.env.get_actions()
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)
        else:
            # Select action leading to state with minimum V(s')
            min_value = float('inf')
            best_actions = []
            for action in actions:
                next_state = self.simulated_step(state, action)
                value = self.v_values[next_state[0], next_state[1]]
                if value < min_value:
                    min_value = value
                    best_actions = [action]
                elif value == min_value:
                    best_actions.append(action)
            return np.random.choice(best_actions)

    def simulated_step(self, state, action):
        # Simulate the environment step without changing the actual environment state
        # Note: Include stochasticity in the simulated step as well
        perpendicular_actions = {
            0: [3, 1],  # Up -> Left, Right
            1: [0, 2],  # Right -> Up, Down
            2: [3, 1],  # Down -> Left, Right
            3: [0, 2]   # Left -> Up, Down
        }

        action_probs = [self.env.correct_action_prob, self.env.perpendicular_prob, self.env.perpendicular_prob]
        actual_actions = [action] + perpendicular_actions[action]
        actual_action = random.choices(actual_actions, weights=action_probs, k=1)[0]

        x, y = state
        if actual_action == 0:  # Up
            x = max(x - 1, 0)
        elif actual_action == 1:  # Right
            y = min(y + 1, self.env.width - 1)
        elif actual_action == 2:  # Down
            x = min(x + 1, self.env.height - 1)
        elif actual_action == 3:  # Left
            y = max(y - 1, 0)

        next_state = (x, y)

        if next_state in self.env.cliff:
            next_state = state  # Agent stays in the same state if it steps into the cliff
        return next_state


    #############################################
    # IMPLEMENT ALGORITHM IN THE TRAIN FUNCTION #
    #############################################
    def train(self, num_episodes):
        pass
        return self.v_values, self.rewards_per_episode

def plot_rewards(rewards_list, labels):
    for rewards, label in zip(rewards_list, labels):
        plt.plot(rewards, label=label)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Episode')
    plt.legend()
    plt.show()

def print_policy_q(q_values, env):
    policy = np.chararray((env.height, env.width), unicode=True)
    actions_symbols = ['↑', '→', '↓', '←']
    for i in range(env.height):
        for j in range(env.width):
            if (i, j) == env.goal_state:
                policy[i, j] = 'G'
            elif (i, j) in env.cliff:
                policy[i, j] = 'C'
            else:
                best_action = np.argmax(q_values[i, j])
                policy[i, j] = actions_symbols[best_action]
    for row in policy:
        print(' '.join(row))

def print_policy_v(v_values, env):
    policy = np.chararray((env.height, env.width), unicode=True)
    actions_symbols = ['↑', '→', '↓', '←']
    actions = env.get_actions()
    td_lambda_agent = TDLambdaAgent(env)
    for i in range(env.height):
        for j in range(env.width):
            state = (i, j)
            if state == env.goal_state:
                policy[i, j] = 'G'
            elif state in env.cliff:
                policy[i, j] = 'C'
            else:
                min_value = float('inf')
                best_actions = []
                for action in actions:
                    next_state = td_lambda_agent.simulated_step(state, action)
                    value = v_values[next_state[0], next_state[1]]
                    if value < min_value:
                        min_value = value
                        best_actions = [action]
                    elif value == min_value:
                        best_actions.append(action)
                best_action = np.random.choice(best_actions)
                policy[i, j] = actions_symbols[best_action]
    for row in policy:
        print(' '.join(row))