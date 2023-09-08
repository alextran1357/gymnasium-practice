import gymnasium as gym
import pandas as pd
import numpy as np

class QLearningTable:
    def __init__(self, observation_space_n, action_space_n, learning_rate=0.2, reward_decay=0.9, e_greedy=0.9):
        self.lr = learning_rate 
        self.gamma = reward_decay # reward decay over time
        self.epsilon = e_greedy # chance of agent picking a random action
        self.q_table = pd.DataFrame(np.zeros((observation_space_n, action_space_n)),
                                    columns=[i for i in range(action_space_n)], 
                                    dtype=np.float64)

    def learn(self, state, next_state, action, reward, terminated , truncated):
        '''
        NEW Q(S, A) = Q(S, A) + learning_rate * (reward(S, A) + (discount_rate * Q(S',A) - Q(S,A_)
        '''
        self.q_table.iloc[state, action] += self.lr * (reward + (self.gamma * (self.q_table.iloc[next_state, :].max() - self.q_table.iloc[state, action])))

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon: # greedy action
            state_action = self.q_table.iloc[state, :] #get all the q-values of this current state
            # a problem with this at the beginning there is only zeros so we need to change this to be random
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else: # exporation action
            action = env.action_space.sample()
        return action


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    Q_table = QLearningTable(env.observation_space.n, env.action_space.n)

    observation, info = env.reset()

    for i in range(10000):
        action = Q_table.choose_action(observation) # choose an action based on the q-table
        next_observation, reward, terminated, truncated, info = env.step(action) # take that action
        Q_table.learn(observation, next_observation, action, reward, terminated, truncated) # learn from the action
        observation = next_observation # update the observation
        print(Q_table.q_table)
        if terminated or truncated:
            # print(reward)
            observation, info = env.reset()

    env.close()
    
