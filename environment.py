import numpy as np
import pandas as pd
from libs.datagen import DataGenerator

class environment():
    def __init__(self, DataGen, normalization) -> None:
        self.datagen = DataGen
        self.window_size = self.datagen.window_size
        self.normalization = normalization

    def action_space(self):
        return [0, 1, 2, 3]

    def observation_space(self):
        return self.day.dtypes
    
    def step(self, action):

        if self.idx < self.len-1:

            self.idx += 1
            done = False
        else:
            done = True

        state = self.day.iloc[self.idx].values
        reward = 1
        
        return state, reward, done

    def reset(self):
        self.day = self.datagen.get_sample(k=1)

        # Normalize to Open of first data
        if self.normalization:
            self.day.loc[:,self.datagen.price_dependent] = self.day.loc[:,self.datagen.price_dependent] -\
                                                           self.day.Open.iloc[self.window_size]
        self.len = len(self.day)  
        self.idx = 0 
        self.action_space_n = 4
        self.observation_space_n = self.day.shape[1]

        # Recording trading activity
        self.tracker = self.day.assign(**{'Critic_Value': np.nan,
                                        'Action': np.nan,
                                        'Current_Profit': np.nan,
                                        'Total_Profit': np.nan,}).iloc[:,-4:]
        self.tracker.insert(loc=0, column='Date', value=self.day.Date)

        # Tradebook for closed trades
        self.trades = pd.DataFrame(columns=['Opentime', 'Closetime','Openindex', 'Closeindex',
                                           'Open', 'Close', 'Profit', 'Direction']) 

        print(f'New Day started: {self.day.Date.iloc[self.window_size]}')


if __name__ == "__main__":
    window_size = 10
    gen = DataGenerator(symbol="SP500_M1",
                        fraction=[1e6, 10000],
                        window_size=window_size)
    env = environment(gen, normalization=False)
    env.reset()
    print(env.day.iloc[0:15,0:10])
    
  
