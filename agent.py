import numpy as np 
from tensorflow.keras.backend import get_value
from model import NN

class Agent:
    def __init__(self, env):
        self.env = env
        self.num_observations = tuple(self.env.observation_space_n)
        self.num_actions = self.env.action_space_n
        self.num_values = 1
        self.game_size =  (self.env.window_size, len(self.env.tracker_list))
        self.gamma = 0.99
        self.epsilon = 0.99
        self.epsilon_decay = 0.9999
        self.epsilon_limit = 0.001
        self.lr_actor = 1e-3
        self.lr_critic = 5e-3
        self.model = NN(self.num_observations, self.num_actions, self.num_values, self.game_size, self.lr_actor, self.lr_critic)

    def get_action(self, state):
        if np.random.rand(1) > self.epsilon:
            policy = self.model.actor(inputs=state, training=False)
            action = np.random.choice(self.num_actions, p=get_value(policy[0]))
        else:
            action = np.random.choice(self.num_actions, p=[0.7,0.1,0.1,0.1])
            if self.epsilon > self.epsilon_limit:
                self.epsilon *= self.epsilon_decay          
        return action

    def get_values(self, state, action, reward, next_state, done):
        values = np.zeros((1, self.num_values))
        advantages = np.zeros((1, self.num_actions))

        value = self.model.predict_critic(state)
        next_value = self.model.predict_critic(next_state)

        if done:
            advantages[0][action] = reward - value
            values[0][0] = reward
        else:
            advantages[0][action] = (reward + self.gamma * next_value) - value
            values[0][0] = reward + self.gamma * next_value

        return advantages, values

    def update_policy(self, states, games, advantages, values):
        self.model.train_actor(states, games, advantages)
        self.model.train_critic(states, games, values)
