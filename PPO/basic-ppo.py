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
    def __init__(self, obs_space, learning_rate=0.01, discount_factor=0.9, epsilon_clip=0.3):
        self.learning_rate = learning_rate
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
        return int(action.numpy()[0])
    
    def calculate_discounted_returns(self, rewards):
        sum_reward = 0
        returns = []
        for r in rewards[::-1]:
            sum_reward = sum_reward * self.gamma + r
            returns.append(sum_reward)
        returns.reverse()
        return returns
    
    def predict_advantage(self, returns, states):
        advantages = []
        value_estimates = [self.critic_model(state) for state in tf.expand_dims(states, 0)]
        value_estimates = tf.reshape(value_estimates, [-1])
        for i in range(len(states)):
            advantage = returns[i] - value_estimates[i]
            advantages.append(advantage)
        return advantages
    
    def compute_advantages(self, states, rewards, values, next_value, gamma=0.99, lambda_=0.95):
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE)
        
        :param rewards: Rewards obtained from the environment, shape: [num_steps,]
        :param values: Values estimated by the critic, shape: [num_steps,]
        :param next_value: Value estimated for the next state, shape: []
        :param dones: Boolean array indicating if an episode is finished at each time step, shape: [num_steps,]
        :param gamma: Discount factor for future rewards, scalar
        :param lambda_: GAE lambda parameter, scalar
        
        :return: advantages and returns, both shaped: [num_steps,]
        """
        
        num_steps = len(rewards)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        values = [self.value_network(state) for state in tf.expand_dims(states, 0)]
        
        future_adv = 0.0
        future_ret = next_value
        for t in reversed(range(num_steps) - 1):
            # Delta represents the temporal difference error
            delta = rewards[t] + gamma * values[t+1] - values[t]
            
            # Compute advantage using GAE
            advantages[t] = delta + gamma * lambda_ * future_adv
            
            # Compute the return
            returns[t] = advantages[t] + values[t]
            
            # Update the future advantage and return for the next iteration
            future_adv = advantages[t]
            future_ret = returns[t]
        
        
EPISODES = 1000
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
    episode_reward = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = agent.select_action(state)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # collect trajectories
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        
        state = next_state
        
    advantages, returns = agent.calculate_advantages_and_returns(trajectories, value_function = V_phi)