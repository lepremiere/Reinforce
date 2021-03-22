import random
import numpy as np
import pandas as pd 
from collections import namedtuple, deque
from operator import attrgetter

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size) -> None:
        
        self.memory = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.batch_size = batch_size 
        self.experience = namedtuple("Experience",
                            field_names=["state", "advantage", "value", "reward"])

    def add(self, state, advantage, value, reward):
        e = self.experience(state, advantage, value, reward)
        self.rewards.append(reward)
        self.memory.append(e)

    def sample(self, skewed=False):
        if skewed:
            weights = np.cumsum(self.rewards/np.sum(self.rewards))
            print(np.shape(weights))
            samples = random.choices(self.memory, cum_weights=weights, k=self.batch_size)
        else:
            samples = random.sample(self.memory, k=self.batch_size)
        
        return samples

    def __len__(self):
        return len(self.memory)

if __name__ =="__main__":
    pass