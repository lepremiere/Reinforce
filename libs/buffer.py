import random
import numpy as np
from multiprocessing import Process

class ReplayBuffer(Process):

    def __init__(self, settings) -> None:
        Process.__init__(self)
        self.memory = []
        self.rewards = []
        self.batch_size = settings['batch_size'] 
        self.buffer_size = settings['buffer_size']

    def add(self, state, advantage, value, reward):
        e = [state, advantage, value, reward]
        self.rewards.append(reward)
        self.memory.append(e)

        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)
            self.rewards.pop(0)
        print(e)

    def get_samples(self, skewed=False):
        if skewed:
            weights = np.cumsum(self.rewards/np.sum(self.rewards))
            samples = random.choices(self.memory, cum_weights=weights, k=self.batch_size)
        else:
            samples = random.sample(self.memory, k=self.batch_size)
        
        return samples

    def __len__(self):
        return len(self.memory)

if __name__ =="__main__":
    pass