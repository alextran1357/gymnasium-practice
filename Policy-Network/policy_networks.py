import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from datetime import datetime

class PolicyNetwork(tf.keras.Model):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=[obs_size])
        self.d1 = tf.keras.layers.Dense(64, activation='relu')
        self.d2 = tf.keras.layers.Dense(32, activation='relu')
        self.out = tf.keras.layers.Dense(action_size,activation='softmax')
    
    def call(self, state):
        x = tf.convert_to_tensor(state)
        x = self.input_layer(state)
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)
        return x

class Agent: 
    def __init__(self, obs_size, action_size, discount_factor=0.9, learning_rate=0.02):
        self.obs_size = obs_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.model = PolicyNetwork(obs_size, action_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def sample_action_from_policy(self, state):
        prob = self.model(np.array([state]))
        dist = tfp.distributions.Categorical(probs=prob)
        action = dist.sample()
        return int(action.numpy()[0])
    
    def log_prob_of_action(self, state, action):
        prob = self.model(np.array([state]), training=True)
        return tfp.distributions.Categorical(probs=prob).log_prob(action)

    
    def train(self, episode_states, episode_actions, episode_rewards):
        episode_rewards.reverse()
        sum_reward = 0
        returns = []
        for r in episode_rewards:
            sum_reward = sum_reward * self.discount_factor + r
            returns.append(sum_reward)
        returns.reverse()
        
        # Calculate the mean and stardard deviation of the returns for normalization
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Normalize the returns
        normalized_returns = [(r - mean_return) / (std_return + 1e-9) for r in returns]
        
        for state, reward, action in zip(episode_states, normalized_returns, episode_actions):
            with tf.GradientTape() as tape:
                log_prob = self.log_prob_of_action(state, action)
                loss = -log_prob * reward
            grads = tape.gradient(loss, self.model.trainable_variables)
            # Apply clipped gradients to prevent exploding gradients
            clipped_grads, _ = tf.clip_by_global_norm(grads, 1.0)
            self.optimizer.apply_gradients(zip(clipped_grads, self.model.trainable_variables))

num_episodes = 1000

# Define the policy network with parameters theta
env = gym.make('CartPole-v0')
obs_size = env.observation_space.shape[0]
action_size = env.action_space.n
CPAgent = Agent(obs_size, action_size)

# Setup tensorboard
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f'logs/{timestamp}'
summary_writer = tf.summary.create_file_writer(log_dir)

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
        action = CPAgent.sample_action_from_policy(state)

        # Take the selected action and observe the next state and reward
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # Store state, action, and reward in the episode trajectory
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)

        # Move to the next state
        state = next_state
        
    CPAgent.train(episode_states, episode_actions, episode_rewards)

    # Print out the total reward for the episode
    with summary_writer.as_default():
        tf.summary.scalar('Total Reward', total_reward, step=episode)
        
    print('Episode: {}, Total reward: {}'.format(episode, total_reward))

# End of training loop
env.close()

# # Initalize another environment to visualize the trained policy
# env = gym.make('CartPole-v1', render_mode='human')
# for episode in range(10):
#     state, info = env.reset()
#     terminate = False
#     truncate = False
#     while not (terminate or truncate):
#         action = PNetwork.sample_action_from_policy(state)
#         next_state, reward, terminated, truncated, info = env.step(action)
#         state = next_state
#         if terminated or truncated:
#             observation, info = env.reset()

# # End of training loop
# env.close()
