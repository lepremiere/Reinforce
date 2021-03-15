import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

class make_environment():
    def __init__(self, df) -> None:
        self.df = df 
        self.len = len(self.df)  
        self.idx = 0 

    def action_spec(self):
        return [0, 1, 2, 3]

    def observation_spec(self):
        return self.df.dtypes
    
    def step(self, action):

        if self.idx < self.len-1:
            self.idx += 1
            done = False
        else:
            print('Last step', self.idx,'/',self.len)
            input('?')
            done = True
            
        return self.df.iloc[self.idx].values, done

    def reset(self):
        self.idx = 0

        return self.df.iloc[0].values, False

    def print(self):
        print(self.df.tail())
