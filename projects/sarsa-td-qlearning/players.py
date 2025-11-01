import numpy as np
import matplotlib.pyplot as plt
import random

def _eta_for(eta, state, t, N_visits):
    if isinstance(eta, (float, int)):
        return float(eta)
    val = float(eta(state, t, N_visits))
    if not np.isfinite(val) or val < 0:
        raise ValueError(f"eta() must return a nonnegative finite float; got {val}")
    return val

class SARSAAgent:
    def __init__(self, env, eta=0.5, gamma=0.9, epsilon=0.1):
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
    def __init__(self, env, eta=0.5, gamma=0.9, epsilon=0.1):
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
    def __init__(self, env, eta=0.5, gamma=0.9, epsilon=0.1, lam=0.9):
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
            max_value = float('-inf')
            best_actions = []
            for action in actions:
                next_state = self.simulated_step(state, action)
                value = self.v_values[next_state[0], next_state[1]]
                # handle infinite values, preventing NaNs
                if not np.isfinite(value):
                    value = float('-inf')
                if value > max_value:
                    max_value = value
                    best_actions = [action]
                elif value == max_value:
                    best_actions.append(action)
            # if no best actions, choose a random action
            if len(best_actions) == 0:
                return np.random.choice(actions)
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

        action_probs = [self.env.prob_correct_action, self.env.prob_left, self.env.prob_right]
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

class OptimalAgent:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.height = env.height
        self.width = env.width
        self.actions = env.get_actions()
        self.perp = {0: [3, 1], 1: [0, 2], 2: [3, 1], 3: [0, 2]}
        self.V = np.zeros((self.height, self.width))
        self.policy = np.zeros((self.height, self.width), dtype=int)

    def _step_det(self, state, actual_action):
        x, y = state
        if actual_action == 0:
            x = max(x - 1, 0)
        elif actual_action == 1:
            y = min(y + 1, self.width - 1)
        elif actual_action == 2:
            x = min(x + 1, self.height - 1)
        elif actual_action == 3:
            y = max(y - 1, 0)
        next_state = (x, y)
        if next_state in self.env.cliff:
            next_state = state
        return next_state

    def value_iteration(self, epsilon=1e-8, max_iter=10000):
        V = np.zeros((self.height, self.width))
        goal = self.env.goal_state
        prob_correct_action, prob_left, prob_right = self.env.prob_correct_action, self.env.prob_left, self.env.prob_right
        
        iterations = 0
        for _ in range(max_iter):
            iterations += 1
            delta = 0.0
            V_new = V.copy()
            for i in range(self.height):
                for j in range(self.width):
                    s = (i, j)
                    if s == goal:
                        V_new[i, j] = 0.0
                        continue
                    if s in self.env.cliff:
                        V_new[i, j] = np.nan
                        continue
                    q_vals = []
                    for a in self.actions:
                        next_states = [
                            self._step_det(s, a),
                            self._step_det(s, self.perp[a][0]),
                            self._step_det(s, self.perp[a][1])
                        ]
                        probs = [prob_correct_action, prob_left, prob_right]
                        # aggregate in case of duplicates
                        exp = 0.0
                        for p, s_next in zip(probs, next_states):
                            r = -1.0
                            v_next = 0.0 if s_next == goal else V[s_next[0], s_next[1]]
                            exp += p * (r + self.gamma * v_next)
                        q_vals.append(exp)
                    V_new[i, j] = max(q_vals)
                    delta = max(delta, abs(V_new[i, j] - V[i, j]))
            V = V_new
            if delta < epsilon:
                break

        if iterations < max_iter:
            print(f"Value iteration converged in {iterations} iterations")
        else:
            print(f"Value iteration did not converge in {max_iter} iterations, outputting last iteration")

        # Derive greedy policy
        pi = np.zeros((self.height, self.width), dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                s = (i, j)
                if s == goal:
                    pi[i, j] = 0
                    continue
                if s in self.env.cliff:
                    pi[i, j] = 0
                    continue
                best_a = 0
                best_q = -1e18
                for a in self.actions:
                    next_states = [
                        self._step_det(s, a),
                        self._step_det(s, self.perp[a][0]),
                        self._step_det(s, self.perp[a][1])
                    ]
                    probs = [prob_correct_action, prob_left, prob_right]
                    q = 0.0
                    for p, s_next in zip(probs, next_states):
                        r = -1.0
                        v_next = 0.0 if s_next == goal else V[s_next[0], s_next[1]]
                        q += p * (r + self.gamma * v_next)
                    if q > best_q:
                        best_q = q
                        best_a = a
                pi[i, j] = best_a
        self.V = V
        self.policy = pi
        return V, pi

    def evaluate(self, num_episodes):
        rewards = []
        discounted_rewards = []
        # Ensure we have a policy
        if np.all(self.V == 0) and (self.env.start_state != self.env.goal_state):
            self.value_iteration()
        for _ in range(num_episodes):
            state = self.env.reset()
            total = 0
            discounted_total = 0.0
            t = 0
            while True:
                a = self.policy[state[0], state[1]]
                next_state, reward, done = self.env.step(a)
                total += reward
                discounted_total += (self.gamma ** t) * reward
                state = next_state
                if done:
                    break
                t += 1
            rewards.append(total)
            discounted_rewards.append(discounted_total)
        # Summarize performance and optionally check against DP value at start
        try:
            avg_reward = float(np.mean(rewards)) if len(rewards) > 0 else float('nan')
            avg_discounted_reward = float(np.mean(discounted_rewards)) if len(discounted_rewards) > 0 else float('nan')
        except Exception:
            avg_reward = float('nan')
            avg_discounted_reward = float('nan')
        start = self.env.start_state
        v_start = self.V[start[0], start[1]]
        # Print both undiscounted and discounted averages
        # print(f"Average total reward over {num_episodes} episodes (undiscounted): {avg_reward:.3f}")
        print(f"Average discounted reward over {num_episodes} episodes (gamma={self.gamma}): {avg_discounted_reward:.3f}")
        if np.isfinite(avg_discounted_reward) and np.isfinite(v_start):
            print(f"DP V(start): {v_start:.3f}")
            print(f"Difference (avg_discounted - V(start)): {avg_discounted_reward - v_start:.3f}")
        else:
            print(f"DP V(start): {v_start}")
        return rewards

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

def print_policy_optimal(policy, env):
    policy_chars = np.chararray((env.height, env.width), unicode=True)
    actions_symbols = ['↑', '→', '↓', '←']
    for i in range(env.height):
        for j in range(env.width):
            state = (i, j)
            if state == env.goal_state:
                policy_chars[i, j] = 'G'
            elif state in env.cliff:
                policy_chars[i, j] = 'C'
            else:
                policy_chars[i, j] = actions_symbols[policy[i, j]]
    for row in policy_chars:
        print(' '.join(row))
