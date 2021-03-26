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
                 val,
                 settings):
        Process.__init__(self)
        self.name = name
        self.window_size = settings['window_size']
        self.normalization = settings['normalization']
        self.skewed = settings['skewed']
        self.verbose = settings['verbose']
        self.data_gen_in_q = data_gen_in_q
        self.data_gen_out_q = data_gen_out_q
        self.env = Environment(in_q=self.data_gen_in_q,
                               out_q=self.data_gen_out_q,
                               news_q=news_q,
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
                states = [[state[0]], [state[1]]]
                rewards = []
                advantages = []
                values = []

                
                while not done:
                    self.batch_gen_in_q.put(('actions', self.name, state, 0))
                    action, _, _ = self.pipe.get()
                    self.pipe.task_done()
                    next_state, reward, done = self.env.step(action=action, epsilon=0)
                    self.batch_gen_in_q.put(('values', self.name, state, (action, reward, next_state, done)))
                    _, advantage, value = self.pipe.get()
                    self.pipe.task_done()
                    
                    advantages.append(advantage)
                    values.append([value])
                    rewards.append([reward])
                    state = next_state
                    if not done:
                        states = (np.concatenate([states[0], [state[0]]], axis=0),
                                  np.concatenate([states[1], [state[1]]], axis=0))
            
                advantages = np.reshape(advantages, np.shape(advantages))
                values = np.reshape(values, np.shape(values))
                self.replay_buffer.add(states, advantages, values, np.mean(rewards))
                
                if self.verbose == 1:
                    # Overwatch 
                    message = ('Episode_end', [len(self.env.day),
                                               round(np.mean(rewards),2),
                                               round(np.min(rewards),2),
                                               round(np.max(rewards),2)])
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

if __name__ == "__main__":
    pass