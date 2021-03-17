import numpy as np 
from tensorflow.keras.backend import get_value
from model import NN
import time

class Agent:
    def __init__(self, env):
        self.env = env
        self.num_observations = self.env.observation_space
        self.num_actions = self.env.action_space
        self.num_values = 1
        self.gamma = 0.99
        self.lr_actor = 1e-3
        self.lr_critic = 5e-3
        self.model = NN(self.num_observations, self.num_actions, self.num_values, self.lr_actor, self.lr_critic)

    def get_action(self, state):
        policy = self.model.actor(state, training=False)
        action = np.random.choice(self.num_actions, p=get_value(policy[0]))
        return action

    def update_policy(self, state, action, reward, next_state, done):
        values = np.zeros((1, self.num_values))
        advantages = np.zeros((1, self.num_actions))

        value = get_value(self.model.critic(state))[0]
        next_value = get_value(self.model.critic(next_state))[0]

        if done:
            advantages[0][action] = reward - value
            values[0][0] = reward
        else:
            advantages[0][action] = (reward + self.gamma * next_value) - value
            values[0][0] = reward + self.gamma * next_value

        self.model.train_actor(state, advantages)
        self.model.train_critic(state, values)

    def train(self, num_episodes):
        total_rewards = []
        t = time.time()
        for episode in range(num_episodes):
            total_reward = 0.0
            state = self.env.reset()

            state = np.reshape(state, (1, self.num_observations))

            while True:
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action)
                next_state = np.reshape(next_state, (1, self.num_observations))

                self.update_policy(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state

                if done:
                    total_rewards.append(total_reward)
                    mean_total_rewards = np.mean(total_rewards[-min(len(total_rewards), 10):])

                    print("Episode: ", episode+1,
                        " Total Reward: ", total_reward,
                        " Mean: ", mean_total_rewards,
                        " Time: ", round(time.time()-t,2), " s")
                    break

        return total_rewards

    def play(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            while True:
                state = np.reshape(state, (1, self.num_observations))
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                if done:
                    break