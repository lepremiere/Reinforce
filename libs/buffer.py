import random
import numpy as np
from multiprocessing import Process

class ReplayBuffer():

    def __init__(self, settings) -> None:
        self.memory = []
        self.rewards = []
        self.batch_size = settings['batch_size'] 
        self.buffer_size = settings['buffer_size']

    def add(self, states, advantages, values, total_reward):
        e = [states, advantages, values]
        self.rewards.append(total_reward)
        self.memory.append(e)

        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)
            self.rewards.pop(0)
        # print(e)

    def get_samples(self, skewed=False):
        if skewed:
            weights = np.cumsum(self.rewards/np.sum(self.rewards))
            samples = random.choices(self.memory, cum_weights=weights, k=1)
        else:
            samples = random.sample(self.memory, k=1)
        
        return samples

    def __len__(self):
        return len(self.memory)

if __name__ =="__main__":
    pass