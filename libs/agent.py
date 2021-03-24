import numpy as np 
from multiprocessing import Process
from tensorflow.keras.backend import get_value
from libs.model import NN

class Agent(Process):

    def __init__(self, env, in_q, out_q, val, settings):   
        Process.__init__(self)
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
        self.val = val
        self.settings = settings
        self.verbose = settings['verbose']

    def run(self):
        self.model = NN(self.num_observations, self.num_actions,
                    self.num_values, self.game_size,
                    self.lr_actor, self.lr_critic,
                    self.settings)
        while True:
            type, ns, states, val_stuff = self.agent_in_q.get()  
            self.agent_in_q.task_done()
            if self.val.value == 0:              
                break

            # Switch between purpose
            if type == 'actions':
                actions = []
                if np.random.rand(1,1) > self.epsilon:
                    policy = self.model.actor(inputs=states, training=False)
                else:
                    policy = [[0.7,0.1,0.1,0.1] for _ in range(len(ns))]
                    if self.epsilon > self.epsilon_limit:
                        self.epsilon *= self.epsilon_decay  
                for i in range(np.shape(policy)[0]):
                    actions.append(np.random.choice(self.num_actions, p=get_value(policy[i])))

                self.distributor_in_q.put((ns, actions, np.zeros(shape=(len(ns))), np.zeros(shape=(len(ns)))))    

            if type == 'values':
                actions, rewards, next_states, dones = val_stuff
                values = np.zeros((len(ns), self.num_values))
                advantages = np.zeros((len(ns), self.num_actions))
                values = get_value(self.model.critic(states, training=False))
                next_values = get_value(self.model.critic(next_states, training=False))

                for i in range(len(dones)):
                    if dones[i]:
                        advantages[i][actions[i]] = rewards[i] - values[i]
                        values[i][0] = rewards[i]
                    else:
                        advantages[i][actions[i]] = (rewards[i] + self.gamma * next_values[i]) - values[i]
                        values[i][0] = rewards[i] + self.gamma * next_values[i]   

                self.distributor_in_q.put((ns, np.zeros(shape=(len(ns))), advantages, values))    

            if type == 'train':
                if self.verbose == 1:
                    print('Training...')
                advantages, values = val_stuff
                self.model.actor.fit(x=states, y=advantages,
                                    epochs=3,
                                    verbose=0)
                self.model.critic.fit(x=states, y=values,
                                     epochs=3,
                                     verbose=0)

        a = self.model.actor.predict(x=(np.array([np.ones(shape=self.num_observations)]), np.array([np.ones(shape=self.game_size)])))
        b = self.model.critic.predict(x=(np.array([np.ones(shape=self.num_observations)]), np.array([np.ones(shape=self.game_size)])))
        self.model.actor.save('./1_model/actor.h5')
        self.model.critic.save('./1_model/critic.h5')
        print('Agent is done!')

if __name__ == "__main__":
    pass