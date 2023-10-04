import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym

class ActorModel(tf.keras.Model):
    def __init__(self, action_size):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(32, activation='relu')
        self.out = tf.keras.layers.Dense(action_size,activation='softmax')
    
    def call(self, state):
        x = self.d1(state)
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

class PPO:
    def __init__(self, action_dim, lr=0.0002, gamma=0.9, epsilon=0.2, lambda_=0.95, c1=1.0, c2=0.01):
        self.action_dim=action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.c1 = c1   
        self.c2 = c2

        self.actor = ActorModel(action_dim)
        self.critic = CriticModel()
        self.actor_optimizer = tf.keras.optimizers.Adam(lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr)

    def get_action(self, state):
        softmax_prob = self.actor(np.array([state]))
        dist = tfp.distributions.Categorical(probs=softmax_prob)
        action = dist.sample()
        action_prob = dist.prob(action)
        return int(action.numpy()[0]), action_prob
    
    def compute_gae(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages
    
    def train(self, states, actions, rewards, next_states, dones):
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        values = tf.squeeze(self.critic(states))
        next_values = tf.squeeze(self.critic(next_states))
        advantages = self.compute_gae(rewards, values.numpy(), next_values.numpy(), dones)
        returns = advantages + values.numpy()

        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            old_probs = self.actor(states)
            old_probs = tf.reduce_sum(old_probs * tf.one_hot(actions, self.action_dim), axis=1)

            # Actor Loss
            new_probs = self.actor(states)
            new_probs = tf.reduce_sum(new_probs * tf.one_hot(actions, self.action_dim), axis=1)
            ratio = new_probs / (old_probs + 1e-10)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Critic Loss
            critic_loss = 0.5 * tf.reduce_mean(tf.square(returns - tf.squeeze(self.critic(states))))

            # Entropy
            entropy = -tf.reduce_mean(self.actor(states) * tf.math.log(self.actor(states) + 1e-10))

            total_loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return total_loss
    
EPISODES = 1000    
SAMPLE_COLLECTION_TIMESTEP = 500
env = gym.make('LunarLander-v2')
agent = PPO(env.action_space.n)

for episode in range(EPISODES):
    state = env.reset()
    state = state[0]
    
    states = []
    rewards = []
    actions = []
    next_states = []
    dones = []
    action_probs = []
    
    for i in range(SAMPLE_COLLECTION_TIMESTEP):
        action, prob = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        action_probs.append(prob)
        next_states.append(next_state)
        if terminated or truncated:
            dones.append(True)
        else:
            dones.append(False)
        
        state = next_state
        
        if terminated or truncated:
            state = env.reset()
            state = state[0]
            
    loss = agent.train(states, actions, rewards, next_states, dones)
    print('Episode: {}, Loss: {}'.format(episode, loss))
    
    
env.close()
        
        
