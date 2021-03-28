import numpy as np 
from multiprocessing import Process
from tensorflow.keras.backend import get_value
from libs.model import NN

class Agent(Process):

    def __init__(self, env, in_q, out_q, news_q, val, settings):   
        Process.__init__(self)
        self.env = env
        self.agent_in_q = in_q
        self.distributor_in_q = out_q
        self.news_q = news_q
        self.num_observations = tuple(self.env.observation_space_n)
        self.num_actions = self.env.action_space_n
        self.num_values = 1
        self.game_size =  (self.env.window_size, len(self.env.tracker_list))
        self.gamma = settings['gamma']
        self.epsilon = settings['epsilon'][0]
        self.temp_epsilon = self.epsilon
        self.epsilon_decay = settings['epsilon'][1]
        self.epsilon_limit = settings['epsilon'][2]
        self.lr_actor = settings['lr_actor']
        self.lr_critic = settings['lr_critic']
        self.best_result = 0
        self.val = val
        self.settings = settings
        self.verbose = settings['verbose']
    
    def get_actions(self, ns, states):
        actions = []
        if np.random.rand(1,1) > self.epsilon:
            policy = self.model.actor(inputs=states, training=False)
        else:
            policy = [[0.6,0.1,0.1,0.2] for _ in range(len(ns))]
            if self.epsilon > self.epsilon_limit:
                self.epsilon *= self.epsilon_decay  
                if self.temp_epsilon - self.epsilon >= 0.01:
                    self.news_q.put(('Epsilon', self.epsilon))
                    self.temp_epsilon = self.epsilon
        for i in range(len(policy)):
            actions.append(np.random.choice(self.num_actions, p=get_value(policy[i])))
        
        return actions
    
    def get_values(self, ns, states, val_stuff):
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

        return advantages, values
    
    def train(self, states, val_stuff):
        advantages, values = val_stuff
        actor_result = self.model.actor.fit(x=states, y=advantages,
                                            epochs=1,
                                            verbose=0)
        critic_result = self.model.critic.fit(x=states, y=values,
                                            epochs=1,
                                            verbose=0)
  
        self.news_q.put(('Loss',[actor_result.history['loss'][-1], critic_result.history['loss'][-1]]))

    def run(self):
        self.model = NN(self.num_observations, self.num_actions,
                    self.num_values, self.game_size,
                    self.lr_actor, self.lr_critic,
                    self.settings)
        if self.verbose > 0:
            print('Agent ready!')
        while True:
            type, ns, states, val_stuff = self.agent_in_q.get()
            self.agent_in_q.task_done()

            if np.sum(self.val[:]) == 0:              
                break

            # Switch between purposes
            if type == 'actions':
                actions = self.get_actions(ns, states)
                self.distributor_in_q.put((ns, actions, np.zeros(shape=(len(ns))), np.zeros(shape=(len(ns)))))    

            if type == 'values':
                advantages, values = self.get_values(ns, states, val_stuff)
                self.distributor_in_q.put((ns, np.zeros(shape=(len(ns))), advantages, values))    

            if type == 'train':
                self.train(states, val_stuff)
                
                # self.model.actor.save('./1_model/actor.h5')
                # self.model.critic.save('./1_model/critic.h5')

        self.model.actor.predict(x=(np.array([np.ones(shape=self.num_observations)]), np.array([np.ones(shape=self.game_size)])))
        self.model.critic.predict(x=(np.array([np.ones(shape=self.num_observations)]), np.array([np.ones(shape=self.game_size)])))
        self.model.actor.save('./1_model/actor.h5')
        self.model.critic.save('./1_model/critic.h5')
        print('Agent is done!')


if __name__ == "__main__":
    pass