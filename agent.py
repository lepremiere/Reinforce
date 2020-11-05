import numpy as np 
import pandas as pd 

class Agent:
    def __init__(self):
        self.action_space = [0,1,-1,2]
        self.positions = {}

    def get_action(self, data):
        P = [1., 0., 0., 0]

        if data['ma7'].iloc[0] < data['ma21'].iloc[0] and data['ma7'].iloc[1] > data['ma21'].iloc[1]:
            P = [0., 1., 0., 0.] # Buy
        elif data['ma7'].iloc[0] > data['ma21'].iloc[0] and data['ma7'].iloc[1] < data['ma21'].iloc[1]:
            P = [0., 0., 1., 0.] # Sell
            
        else:
            P = [1., 0., 0., 0.]
        
        return 1