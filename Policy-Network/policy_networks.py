import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class initialize_policy_network: 
    def __init__(self, obs_size, action_size, discount_factor=0.99, learning_rate=0.01):
        self.obs_size = obs_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.model = self._create_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def sample_action_from_policy(self, state):
        softmax_output = self.model(np.array([state]))
        action_choice = tfp.distributions.Categorical(probs=softmax_output, dtype=tf.float32).sample()
        return int(action_choice.numpy()[0])
    
    def calculate_returns(self, rewards):
        rewards.reverse()
        sum_reward = 0
        returns = []
        for i in range(len(rewards)):
            sum_reward = sum_reward * self.discount_factor + rewards[i]
            returns.append(sum_reward)
        returns.reverse()
        return returns
    
    def log_prob_of_action(self, state, action):
        softmax_output = self.model(np.array([state]), training=True)
        return tfp.distributions.Categorical(probs=softmax_output, dtype=tf.float32).log_prob(action)
    
    def update_policy_parameters(self, episode_states, episode_actions, returns):
        for state, rewards, actions in zip(episode_states, returns, episode_actions):
            with tf.GradientTape() as tape:
                log_prob = self.log_prob_of_action(state, actions)
                loss = -log_prob * rewards
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def _create_model(self):
        network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=[self.obs_size]),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])
        
        return network

num_episodes = 1000

# Define the policy network with parameters theta
env = gym.make('CartPole-v1')
obs_size = env.observation_space.shape[0]
action_size = env.action_space.n
PNetwork = initialize_policy_network(obs_size, action_size)

# Initialize an empty list to store trajectory information
trajectories = []

# Set hyperparameters
learning_rate = 0.01
discount_factor = 0.99

# Training loop
for episode in range(num_episodes):
    # Initialize episode-specific variables
    episode_states = []
    episode_actions = []
    episode_rewards = []

    # Reset the environment to the initial state
    total_reward = 0
    state, info = env.reset()
    terminated = False
    truncated = False
    # Generate an episode by interacting with the environment
    while not (terminated or truncated):  # Continue until the episode is done
        # Sample an action from the policy
        action = PNetwork.sample_action_from_policy(state)

        # Take the selected action and observe the next state and reward
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # Store state, action, and reward in the episode trajectory
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)

        # Move to the next state
        state = next_state

    # Calculate the return (sum of rewards) for each time step in the episode
    returns = PNetwork.calculate_returns(episode_rewards)

    # Update the policy network's parameters using gradient ascent
    PNetwork.update_policy_parameters(episode_states, episode_actions, returns)

    # Store the trajectory information for later analysis
    trajectories.append((episode_states, episode_actions, episode_rewards))

    # Print out the total reward for the episode
    print('Episode: {}, Total reward: {}'.format(episode, total_reward))

# End of training loop
env.close()

# Initalize another environment to visualize the trained policy
env = gym.make('CartPole-v1', render_mode='human')
for episode in range(10):
    state, info = env.reset()
    terminate = False
    truncate = False
    while not (terminate or truncate):
        action = PNetwork.sample_action_from_policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        if terminated or truncated:
            observation, info = env.reset()

# End of training loop
env.close()
