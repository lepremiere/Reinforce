import numpy as np 
import pandas as pd 

class Agent:
    def __init__(self):
        pass

    def get_action(self, state):     
        return np.random.choice([0,1,2,3], p=[0.3,0.3,0.0,0.4])