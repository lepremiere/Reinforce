import random
import pandas as pd 
from collections import namedtuple, deque
from operator import attrgetter

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size) -> None:
        
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size 
        self.experience = namedtuple("Experience",
                                     field_names=["state", "game", "advantage", "value", "reward"])

    def add(self, state, game, advantage, value, reward):
        e = self.experience(state, game, advantage, value, reward)
        self.memory.append(e)

    def sample(self):
        self.memory = sorted(self.memory, key=attrgetter('reward'), reverse=True)
        return random.sample(self.memory[0:1000], k=self.batch_size)

    def __len__(self):
        return len(self.memory)