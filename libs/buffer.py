import random
import pandas as pd 
from collections import namedtuple, deque

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size) -> None:
        
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size 
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        return len(self.memory)