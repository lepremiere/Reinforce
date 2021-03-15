import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

class make_environment():
    def __init__(self, df) -> None:
        self.df = df    

    def action_spec(self):
        pass 

    def observation_spec(self):
        pass
    
    def step(self):
        pass

    def reset(self):
        pass

    def print(self):
        print(self.df.tail())
