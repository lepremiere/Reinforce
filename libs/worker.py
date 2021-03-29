import sys
import time
import numpy as np
from multiprocessing import Process

from tensorflow.python.keras.layers.merge import concatenate

from libs.environment import Environment
from libs.datagen import DataGenerator
from libs.buffer import ReplayBuffer

class Worker(Process):
    
    def __init__(self, name,
                 data_gen_in_q, data_gen_out_q,
                 task_q,
                 agent_in_q,
                 batch_gen_in_q,
                 pipe,
                 replay_buffer,
                 news_q,
                 market,
                 val,
                 settings):
        Process.__init__(self)
        self.name = name
        self.window_size = settings['window_size']
        self.normalization = settings['normalization']
        self.gamma = settings['gamma']
        self.skewed = settings['skewed']
        self.verbose = settings['verbose']
        self.data_gen_in_q = data_gen_in_q
        self.data_gen_out_q = data_gen_out_q
        self.env = Environment(in_q=self.data_gen_in_q,
                               out_q=self.data_gen_out_q,
                               news_q=news_q,
                               market=market,
                               settings=settings)
        self.val = val

        # Queues and Pipe
        self.task_q = task_q
        self.news_q = news_q
        self.agent_in_q = agent_in_q
        self.batch_gen_in_q = batch_gen_in_q
        self.replay_buffer = replay_buffer
        self.pipe = pipe

    def run(self):
        while True:
            next_task = self.task_q.get()
            self.task_q.task_done()

            if next_task is None:
                break
            
            if next_task == 'play':
                state = self.env.reset()
                done = False
                actions = []
                states1 = []
                states2 = []
                next_states = []
                rewards = []
                advantages = []
                values = []
                dones = []
                
                while not done:
                    self.batch_gen_in_q.put(('play', self.name, [np.expand_dims(state[0], axis=2), np.expand_dims(state[1], axis=2)], 0))
                    action, value = self.pipe.get()
                    self.pipe.task_done()
                    next_state, reward, done = self.env.step(action=action, epsilon=0)
                    
                    actions.append(action)
                    rewards.append([reward])
                    values.append(value)
                    dones.append(done)
                    next_states.append(next_state)
                    states1.append(state[0])
                    states2.append(state[1])
                    state = next_state

                # Advantages, Values
                discounted_rewards = [np.sum(0.95**np.arange(len(rewards[i:]))*rewards[i:]) for i in range(len(rewards))]
                # discounted_rewards = rewards
                advantages = np.zeros(shape=(len(dones), 4))
                for i in range(len(dones)):
                    if dones[i]:
                        advantages[i][actions[i]] = discounted_rewards[i] - values[i]
                        values[i][0] = discounted_rewards[i]
                    else:
                        advantages[i][actions[i]] = (discounted_rewards[i] + self.gamma * values[i+1]) - values[i]
                        values[i][0] = discounted_rewards[i] + self.gamma * values[i+1]  

                advantages = np.reshape(advantages, np.shape(advantages))
                values = np.reshape(values, np.shape(values))
                states1 = np.expand_dims(states1, axis=3)
                states2 = np.expand_dims(states2, axis=3)
                # for i in range(len(dones)):
                #     if 
                self.replay_buffer.add([states1, states2], advantages, values, np.mean(discounted_rewards))
                
                if self.verbose == 1:
                    # Overwatch 
                    message = ('Episode_end', [len(self.env.day),
                                               round(np.mean(rewards),2),
                                               round(np.min(rewards),2),
                                               round(np.max(rewards),2),
                                               round(np.sum(self.env.trades['Profit']), 2),
                                               round(np.mean(self.env.trades['Duration']), 2),
                                               round(len(self.env.trades))])
                    self.news_q.put(message)

            elif next_task == 'train':
         
                if self.verbose == 1:
                    print('Training...')
                samples = self.replay_buffer.get_samples(skewed=self.skewed)
                states, advantages, values = samples[0]
                self.batch_gen_in_q.put(('train', self.name, states, (advantages, values)))
             

        print('Worker:', self.name,' is done!')
        self.val[int(self.name)] = 0
    
    def get_statistics(self):
        pass

    def discount_reward(self, rewards):
        discounted_reward = []
        for i in range(len(rewards)):
            actual = rewards[i]
            gammas = 0.9**np.arange(1, len(rewards[i:])+1)
            discounted_reward.append(rewards[i:]@gammas)
        
        return discounted_reward

if __name__ == "__main__":
    pass