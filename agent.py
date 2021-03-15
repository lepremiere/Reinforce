import numpy as np 
import pandas as pd 

class Agent:
    def __init__(self, env):
        self.action_space = env.action_spec()
        self.obs_space = env.observation_spec()

    def get_action(self, state):     
        return np.random.choice([0,1,2,3])