import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym
import numpy as np

class ActorModel(tf.keras.Model):
    def __init__(self, action_size):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(32, activation='relu')
        self.out = tf.keras.layers.Dense(action_size,activation='softmax')
    
    def call(self, state):
        x = tf.convert_to_tensor(state)
        x = self.d1(x)
        x = self.out(x)
        return x

class CriticModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(32, activation='relu')
        self.out = tf.keras.layers.Dense(1)
    
    def call(self, state):
        x = self.d1(state)
        x = self.out(x)
        return x

class Agent():
    def __init__(self, discount_factor = 0.9):
        self.actor_model = ActorModel()
        self.critic_model = CriticModel()
        self.gamma = discount_factor
    
    def select_action(self, state):
        softmax_prob = self.actor_model(np.array([state]))
        dist = tfp.distributions.Categorical(prob=softmax_prob)
        action = dist.sample()
        return int(action.numpy()[0])
    
    def calculate_discounted_returns(self, rewards):
        sum_reward = 0
        returns = []
        for r in rewards[::-1]:
            sum_reward = sum_reward * self.gamma + r
            returns.append(sum_reward)
        returns.reverse()
    
    def predict_advantage(self, states):
        pass
    
    def update_critic(self, states, returns, learning_rate_critic):
        pass
    
    def update_actor(self, states, actions, advantages, learning_rate_actor):
        pass


env = gym.make('LunarLander-v2',
               continuous=False,
               render_mode='human')
agent = Agent()

# Set hyperparameters
learning_rate_actor = ...
learning_rate_critic = ...
discount_factor = ...
num_episodes = ...

for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    # Initialize episode-specific lists to store states, actions, and rewards
    states = []
    actions = []
    rewards = []
    
    while not done:
        # Actor selects an action based on the current policy
        action = agent.select_action(state)
        
        # Take the selected action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)
        
        # Store the current state, action, and reward
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
    
    # Calculate the discounted returns for each time step
    returns = agent.calculate_discounted_returns(rewards, discount_factor)
    
    # Update the critic network using the returns as targets
    agent.update_critic(states, returns, learning_rate_critic)
    
    # Compute the advantage estimates (returns - value estimates) for the actor update
    advantages = returns - agent.predict_advantage(states)
    
    # Update the actor network using policy gradients with advantages as weights
    agent.update_actor(states, actions, advantages, learning_rate_actor)
