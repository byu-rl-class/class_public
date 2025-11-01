import numpy as np
import matplotlib.pyplot as plt
import random
from gridworld import *
from players import *


env = GridWorld()
num_episodes = 500

# SARSA Agent
sarsa_agent = SARSAAgent(env)
sarsa_q_values, sarsa_rewards = sarsa_agent.train(num_episodes)

# Q-Learning Agent
q_learning_agent = QLearningAgent(env)
q_learning_q_values, q_learning_rewards = q_learning_agent.train(num_episodes)

# TD(λ) Agent
# eta_func = lambda x, t, N_visits: 0.9 / (1 + N_visits[x])
td_lambda_agent = TDLambdaAgent(env, lam=0.9, eta=eta_func, gamma=0.9)
td_lambda_v_values, td_lambda_rewards = td_lambda_agent.train(num_episodes)

# Optimal Agent (planning)
optimal_agent = OptimalAgent(env)
V_opt, pi_opt = optimal_agent.value_iteration()
_ = optimal_agent.evaluate(num_episodes)

# Plotting the rewards
plot_rewards(
    [sarsa_rewards, q_learning_rewards, td_lambda_rewards],
    ['SARSA', 'Q-Learning', f'TD(λ), λ={td_lambda_agent.lam}']
)

# Print Policies
print("Policy derived from SARSA:")
print_policy_q(sarsa_q_values, env)
print("\nPolicy derived from Q-Learning:")
print_policy_q(q_learning_q_values, env)
print("\nPolicy derived from TD(λ):")
print_policy_from_policy(TDLambdaAgent.policy, env)

print("\nOptimal policy (from value iteration):")
print_policy_from_policy(pi_opt, env)