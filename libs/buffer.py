import random
import numpy as np
from multiprocessing import Process
from collections import namedtuple, deque

class ReplayBuffer(Process):

    def __init__(self, settings) -> None:
        Process.__init__(self)
        self.memory = deque(maxlen=settings['buffer_size'])
        self.rewards = deque(maxlen=settings['buffer_size'])
        self.batch_size = settings['batch_size'] 
        self.experience = namedtuple("Experience",
                            field_names=["state", "advantage", "value", "reward"])

    def add(self, state, advantage, value, reward):
        e = self.experience(state, advantage, value, reward)
        self.rewards.append(reward)
        self.memory.append(e)

    def sample(self, skewed=False):
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