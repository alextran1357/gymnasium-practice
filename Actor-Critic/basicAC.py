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
    def __init__(self, action_size, discount_factor=0.9, actor_learning_rate=0.01, critic_learning_rate=0.01):
        self.gamma = discount_factor
        self.actor_model = ActorModel(action_size)
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
        self.critic_model = CriticModel()
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)
        self.critic_loss = tf.keras.losses.MeanSquaredError()
    
    def select_action(self, state):
        softmax_prob = self.actor_model(np.array([state]))
        dist = tfp.distributions.Categorical(probs=softmax_prob)
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
    
    def update_critic(self, states, returns):
        with tf.GradientTape() as tape:
            predicted_values = self.critic_model(tf.expand_dims(states, 0))
            loss = self.critic_loss(returns, predicted_values)
        critic_gradient = tape.gradient(loss, self.critic_model.trainable_variables)  
        self.critic_opt.apply_gradients(zip(critic_gradient, self.critic_model.trainable_variables))
    
    def update_actor(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            action_pred = self.actor_model(tf.expand_dims(states, 0))
            # Grab the indicies of the actions taken at each action_prediction
            action_indices = tf.range(0, tf.shape(action_pred)[0]) * tf.shape(action_pred)[1] + actions
            # Select the action from the index list
            chosen_action_probs = tf.gather(tf.reshape(action_pred, [-1]), action_indices)
            actor_loss = -tf.reduce_mean(tf.math.log(chosen_action_probs) * advantages)
        actor_gradient = tape.gradient(actor_loss, self.actor_model.trainable_variables)  
        self.actor_opt.apply_gradients(zip(actor_gradient, self.actor_model.trainable_variables))


env = gym.make('LunarLander-v2',
               continuous=False)
action_size = env.action_space.n
agent = Agent(action_size)
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    state = state[0]
    
    states = []
    actions = []
    rewards = []
    episode_reward = 0
    terminated = False
    truncated =  False
    
    while not (terminated or truncated):
        # Actor selects an action based on the current policy
        action = agent.select_action(state)
        
        # Take the selected action and observe the next state and reward
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Store the current state, action, and reward
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        episode_reward+= reward
        state = next_state
    
    # Calculate the discounted returns for each time step
    returns = agent.calculate_discounted_returns(rewards)
    
    # Update the critic network using the returns as targets
    agent.update_critic(states, returns)
    
    # Compute the advantage estimates (returns - value estimates) for the actor update
    advantages = agent.predict_advantage(returns, states)
    
    # Update the actor network using policy gradients with advantages as weights
    agent.update_actor(states, actions, advantages)
