import gymnasium as gym
from collections import deque
import tensorflow as tf
import numpy as np
import random

MEMORY_SIZE = 500
EPISODES = 10
BATCH_SIZE = 32
UPDATE_TARGET = 20

class Agent:
    def __init__(self, obs_size, action_size, memory_size, batch_size, greedy_epsilon=0.9, learning_rate=0.01, gamma=0.95):
        self.epsilon = greedy_epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma

        # Initalize action-value function Q with random weights theta
        self.main_network = self._build_network(obs_size, action_size)
        
        # Initalize target action-value function Q with weights theta' = theta
        self.target_network = self._build_network(obs_size, action_size)
        self.target_network.set_weights(self.main_network.get_weights())

        # Initalize memory D with size MEMORY_SIZE
        self.memory_size=memory_size
        self.memory_bank = deque()

    def _build_network(self,obs_size, action_size):
        network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=[obs_size]),
            tf.keras.layers.Dense(24, activation='linear'),
            tf.keras.layers.Dense(action_size, activation='linear')
        ])
        network.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return network
    
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def train(self):
        if len(self.memory_bank) < self.batch_size:
            return 0
        
        mini_batch = random.sample(self.memory_bank, self.batch_size)
        states, actions, rewards, next_states, terminates, truncates = list(zip(*mini_batch))

        target = self.main_network.predict(np.array(states).squeeze(), verbose=0)
        next_state_vals = self.target_network.predict(np.array(next_states).squeeze(), verbose=0)

        max_actions = np.argmax(next_state_vals, axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        target[batch_index, actions] = rewards + (self.learning_rate * next_state_vals[batch_index, max_actions] * terminates * truncates)
        self.target_network.fit(np.array(states).squeeze(), target, verbose=0)


    def choose_action(self, state):
        if np.random.uniform(low=0, high=1) > .5: # random action
            return env.action_space.sample()
        else:
            return np.argmax(self.main_network.predict(state, verbose=0))

    def store_memory(self, state, action, reward, next_state, terminated, truncated):
        if len(self.memory_bank) < self.memory_size:
            self.memory_bank.append((state, action, reward, next_state, terminated, truncated))
        else:
            self.memory_bank.popleft()
            self.memory_bank.append((state, action, reward, next_state, terminated, truncated))


if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initalize Agent
    DQNAgent = Agent(state_size, action_size, MEMORY_SIZE, BATCH_SIZE, UPDATE_TARGET) 

    for i in range(EPISODES):
        # Initialize sequence
        terminated = False
        truncated = False
        total_reward = 0
        time_step = 0

        observation, _ = env.reset()
        observation = np.reshape(observation, [1, state_size])

        while not terminated or truncated:

            # with probability epsilon select a random action
            # otherwise select argmax of main function
            action = DQNAgent.choose_action(np.array(observation))

            # execute action (action) in emulator and observe reward r(t) and image X(t+1)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = np.reshape(next_obs, [1, state_size])
            # store transition
            DQNAgent.store_memory(observation, action, reward, next_obs, terminated, truncated)
            #train
            DQNAgent.train()
            # update observation
            observation = next_obs
           
            total_reward += reward

            time_step += 1
            if time_step % UPDATE_TARGET == 0:
                DQNAgent.update_target_network()

            if terminated or truncated:
                print(total_reward)



    env.close()