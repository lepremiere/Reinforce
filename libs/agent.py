import numpy as np 
from multiprocessing import Process
from tensorflow.keras.backend import get_value
from libs.model import NN

class Agent():

    def __init__(self, env, in_q, out_q, settings):   
        self.env = env
        self.agent_in_q = in_q
        self.distributor_in_q = out_q
        self.num_observations = tuple(self.env.observation_space_n)
        self.num_actions = self.env.action_space_n
        self.num_values = 1
        self.game_size =  (self.env.window_size, len(self.env.tracker_list))
        self.gamma = 0.99
        self.epsilon = 0.001
        self.epsilon_decay = 0.999
        self.epsilon_limit = 0.001
        self.lr_actor = 1e-5
        self.lr_critic = 5e-5
        self.model = NN(self.num_observations, self.num_actions,
                         self.num_values, self.game_size,
                         self.lr_actor, self.lr_critic,
                         settings)

    def run(self):
        while True:
            ns, states1, states2 = self.agent_in_q.get()
            print(np.shape(states1),np.shape(states2))
            actions = []
            if states1 == None:
                break
            if np.random.rand(1,1) > self.epsilon:
                policy = self.model.actor(inputs=(states1, states2), training=False)
                print(policy[0])
            else:
                policy = [[0.7,0.1,0.1,0.1] for _ in range(len(ns))]
                if self.epsilon > self.epsilon_limit:
                    self.epsilon *= self.epsilon_decay  #
            print(policy)
            for i in range(np.shape(policy)[0]):
                actions.append(np.random.choice(self.num_actions, p=get_value(policy[i])))
            self.distributor_in_q.put((ns, actions))   
            self.agent_in_q.task_done()   

    # def get_values(self, state, action, reward, next_state, done):
    #     values = np.zeros((1, self.num_values))
    #     advantages = np.zeros((1, self.num_actions))

    #     value = self.model.predict_critic(state)
    #     next_value = self.model.predict_critic(next_state)

    #     if done:
    #         advantages[0][action] = reward - value
    #         values[0][0] = reward
    #     else:
    #         advantages[0][action] = (reward + self.gamma * next_value) - value
    #         values[0][0] = reward + self.gamma * next_value

    #     return advantages, values

    # def update_policy(self, states, advantages, values):
    #     self.model.train_actor(states, advantages)
    #     self.model.train_critic(states, values)

if __name__ == "__main__":
    pass