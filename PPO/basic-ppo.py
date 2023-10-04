# Initialize policy parameters theta
# Initialize value function parameters phi
# Initialize optimization algorithm hyperparameters (e.g., learning rate, epsilon_clip)

# for iteration = 1, 2, ..., N do
#     # Collect trajectories using the current policy
#     trajectories = collect_trajectories(policy = pi_theta)
    
#     # Calculate advantages and returns for each timestep in the trajectories
#     advantages, returns = calculate_advantages_and_returns(trajectories, value_function = V_phi)
    
#     for optimization_step = 1, 2, ..., K do
#         for trajectory in trajectories do
#             for t = 1, 2, ..., T do
#                 # Calculate the probability ratio
#                 rho_t = pi_theta(trajectory.actions[t] | trajectory.states[t]) / pi_theta_old(trajectory.actions[t] | trajectory.states[t])
                
#                 # Calculate the clipped surrogate objective
#                 L_clip = min(rho_t * advantages[t], clip(rho_t, 1 - epsilon_clip, 1 + epsilon_clip) * advantages[t])
                
#                 # Optional: Add a value function loss term and an entropy regularization term
#                 # L = L_clip - c1 * value_function_loss + c2 * entropy_bonus
                
#                 # Calculate gradients of the objective w.r.t. policy parameters
#                 gradients = compute_gradients(L_clip, w.r.t. = theta)
                
#                 # Update policy parameters
#                 theta = optimize_policy(theta, learning_rate, gradients)
                
#                 # Optional: Update the value function parameters
#                 # phi = optimize_value_function(returns, V_phi, states, learning_rate_vf)
                
#                 # Ensure numerical stability
#                 theta, phi = enforce_numerical_stability(theta, phi)
#             end for
#         end for
#     end for
# end for

# STILL WIP ---------------------------------------------------------------------------------

import tensorflow as tf
import gymnasium as gym
import tensorflow_probability as tfp
import numpy as np

class PolicyNetwork(tf.keras.Model):
    def __init__(self, obs_space):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(32, activation='relu')
        self.out = tf.keras.layers.Dense(obs_space, activation='softmax')
    
    def call(self, state):
        x = self.d1(state)
        x = self.out(x)
        return x
    
class ValueNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(32, activation='relu')
        self.out = tf.keras.layers.Dense(1)
    
    def call(self, state):
        x = self.d1(state)
        x = self.out(x)
        return x
    
class Agent():
    def __init__(self, obs_space, learning_rate=0.01, discount_factor=0.9, epsilon_clip=0.3, lambda_=0.95):
        self.lambda_ = lambda_
        self.gamma = discount_factor
        self.epsilon_clip = epsilon_clip
        self.value_network = ValueNetwork()
        self.policy_network = PolicyNetwork(obs_space)
        self.value_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.policy_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
    def select_action(self, state):
        probs = self.policy_network(state)
        dist = tfp.distributions.Categorical(probs=probs)
        action = dist.sample()
        return int(action.numpy()[0]), probs[0]
    
    def compute_advantages(self, states, rewards, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_advantage = 0
        values = [self.value_network(state) for state in tf.expand_dims(states, 0)]
        running_return = values[-1]
        
        for t in reversed(range(len(rewards))):
            if t != len(rewards) - 1:
                bootstrap_value = values[t+1]
            else:
                bootstrap_value = 0
                
            delta = rewards[t] + self.gamma * bootstrap_value * (1-dones[t]) - values[t]
            running_advantage = delta + self.gamma * self.lambda_ * running_advantage * (1 - dones[t])
            advantages[t] = running_advantage
            returns[t] = running_return
        
        return advantages, returns
        
EPISODES = 1000
DATA_COLLECTION_TIMESTEP = 1000

env = gym.make('LunarLander-v2',
               continuous=False)
obs_space = env.observation_space.n
agent = Agent()

for i in range(EPISODES):
    state = env.reset()
    
    # Collect trajectories
    states = []
    rewards = []
    actions = []
    dones = []
    old_probs = []
    values = []
    
    episode_reward = 0
    terminated = False
    truncated = False
    
    for i in range(DATA_COLLECTION_TIMESTEP):
        action, prob = agent.select_action(state)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # collect trajectories
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        if terminated or truncated:
            dones.append(True)
        else:
            dones.append(False)
        old_probs.append(prob)
        
        state = next_state

        if terminated or truncated:
            state = env.reset()
        
    # Add value estimate for the state after the last state in the batch.
    values = [agent.value_network(state) for state in tf.expand_dims(states, 0)]
    next_value = agent.value_network(np.array([state]))
    values.append(next_value)
    
    compute_advantages(states, rewards, values, dones)