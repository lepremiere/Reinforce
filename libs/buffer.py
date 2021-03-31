import random
import numpy as np
from multiprocessing import Process

## As Process biatch!
class ReplayBuffer(Process):

    def __init__(self, in_q, out_q, news_q, val, settings) -> None:
        Process.__init__(self)
        self.in_q = in_q
        self.out_q = out_q
        self.news_q = news_q
        self.val = val
        self.memory = []
        self.rewards = []
        self.batch_size = settings['buffer_batch_size'] 
        self.buffer_size = settings['buffer_size']
        self.skewed_memory = settings['skewed_memory']
        self.sort_buffer = settings['sort_buffer']

    def add_sample(self, states, advantages, values, total_reward):
        e = [states, advantages, values]
        self.rewards.append(total_reward)
        self.memory.append(e)

        # if self.sort_buffer:
        #     idx = np.argsort(self.rewards)
        #     self.rewards = self.rewards[idx]
        #     self.memory = self.memory[idx]

        if len(self.rewards) > self.buffer_size:
            e = self.memory.pop(0)
            e = self.rewards.pop(0)

    def get_samples(self):
        if self.skewed_memory:
            weights = np.cumsum(self.rewards/np.sum(self.rewards))
            samples = random.choices(self.memory, cum_weights=weights, k=self.batch_size)
        else:
            samples = random.sample(self.memory, k=self.batch_size)
        
        return samples

    def __len__(self):
        return len(self.memory)

    def run(self):
        while True:
            try:
                task, params = self.in_q.get(timeout=0.1)
                self.in_q.task_done()
                if task == 'get_samples':
                    samples = self.get_samples()
                    self.out_q.put(samples)
                elif task == 'add_sample':
                    states, advantages, values, total_reward = params
                    self.add_sample(states, advantages, values, total_reward)
                    del states
                    del advantages
                    del values
                    del total_reward
                    del samples
            except: 
                if np.sum(self.val[:]) == 0:
                    break
                pass

if __name__ =="__main__":
    pass