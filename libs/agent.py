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
        self.num_observations = (self.env.observation_space[0], self.env.observation_space[1])
        self.num_actions = self.env.action_space_n
        self.num_values = 1
        self.game_size =  (len(self.env.tracker_list),)
        self.gamma = settings['gamma']
        self.epsilon = settings['epsilon'][0]
        self.temp_epsilon = self.epsilon
        self.epsilon_decay = settings['epsilon'][1]
        self.epsilon_limit = settings['epsilon'][2]
        self.lr_actor = settings['lr_actor']
        self.lr_critic = settings['lr_critic']
        self.epochs = settings['training_epochs']
        self.best_result = 0
        self.val = val
        self.settings = settings
        self.verbose = settings['verbose']
    
    def get_actions(self, ns, states):
        actions = []
        if np.random.rand(1,1) > self.epsilon:
            policy = self.model.actor(inputs=states, training=False)
        else:
            policy = [[0.25,0.25,0.25,0.25] for _ in range(len(ns))]
            if self.epsilon > self.epsilon_limit:
                self.epsilon *= self.epsilon_decay  
                if self.temp_epsilon - self.epsilon >= 0.01:
                    self.news_q.put(('Epsilon', self.epsilon))
                    self.temp_epsilon = self.epsilon
        for i in range(len(policy)):
            actions.append(np.random.choice(self.num_actions, p=get_value(policy[i])))
        
        return actions
    
    def get_values(self, states):
        values = get_value(self.model.critic(states, training=False))
        return  values
    
    def train(self, states, val_stuff):
        advantages, values = val_stuff
        actor_result = self.model.actor.fit(x=states, y=advantages,
                                            epochs=self.epochs,
                                            verbose=0)
        critic_result = self.model.critic.fit(x=states, y=values,
                                            epochs=self.epochs,
                                            verbose=0)
        self.news_q.put(('Loss',[actor_result.history['loss'][-1], critic_result.history['loss'][-1]]))
        del advantages
        del states
        del val_stuff

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
            if type == 'play':
                actions = self.get_actions(ns, states)
                values = self.get_values(states)
                self.distributor_in_q.put((ns, actions, values))
                del actions
                del values
                del ns

            if type == 'train':
                self.train(states, val_stuff)
                del states
                del val_stuff
                # self.model.actor.save('./1_model/actor.h5')
                # self.model.critic.save('./1_model/critic.h5')

        self.model.actor.predict(x=(np.array([np.ones(shape=self.num_observations)]), np.array([np.ones(shape=self.game_size)])))
        self.model.critic.predict(x=(np.array([np.ones(shape=self.num_observations)]), np.array([np.ones(shape=self.game_size)])))
        self.model.actor.save('./1_model/actor.h5')
        self.model.critic.save('./1_model/critic.h5')
        print('Agent is done!')


if __name__ == "__main__":
    pass