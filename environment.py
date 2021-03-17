import numpy as np
import time
from numpy.core.numeric import NaN
import pandas as pd
from libs.datagen import DataGenerator

class environment():
    def __init__(self, DataGen, normalization, verbose=0) -> None:
        self.verbose = verbose
        self.datagen = DataGen
        self.window_size = self.datagen.window_size
        self.normalization = normalization
        self.observation_space = self.datagen.df.dtypes
        self.action_space = [0,1,2,3]

####################################################################################    
    def step(self, action):

        state = self.day[self.idx:self.window_size + self.idx, 1:]

        # Terminal state 
        if self.idx < self.len-1:
            if action == 0:
                pass
            elif action == 3 and self.trade['Opentime'] != None:
                self.close_position()
            elif action == 1 and self.trade['Opentime'] == None:
                self.open_position(action) 
            self.idx += 1
            done = False
        else:
            done = True

        reward = 1
        
        return state, reward, done
####################################################################################   
    def open_position(self, action):
        # 8 = Close in df. 4 = Open price in tradebook
        direction = action < 2
        self.trade['Opentime'] = self.day[self.idx, 0]
        self.trade['Openindex'] = self.idx
        self.trade['Open'] = self.day[self.idx, 8]
        self.trade['Direction'] = (-1 + 2*direction)[0]
        # print('Opened')
####################################################################################    
    def close_position(self):
        self.trade['Closetime'] = self.day[self.idx, 0]
        self.trade['Closeindex'] = self.idx
        self.trade['Close'] = self.day[self.idx, 8]
        self.trade['Profit'] = (self.trade['Close'] -\
                                self.trade['Open']) *\
                                self.trade['Direction']
        self.trades = self.trades.append(self.trade, ignore_index=True) 
        if self.verbose > 1:   
            print(f'Opentime: {self.trade["Opentime"]}  Closetime: {self.trade["Closetime"]}  ', 
                    f'Open: {round(self.trade["Open"],2)}  Close: {round(self.trade["Close"],2)}  Profit: {round(self.trade["Profit"],2)}')
        self.trade = self.trade_template.copy()
        
####################################################################################
    def reset(self):
        self.day = self.datagen.get_sample(k=1)

        # Normalize to Open of first data
        if self.normalization:
            self.day.loc[:,self.datagen.price_dependent] = self.day.loc[:,self.datagen.price_dependent] -\
                                                           self.day.Open.iloc[self.window_size]
        # Params
        self.action_space_n = 4
        self.observation_space_n = self.day.shape[1]
        self.len = len(self.day) - self.window_size
        self.idx = 0 

        # Recording trading activity
        self.tracker = self.day.assign(**{'Critic_Value': np.nan,
                                          'Action': np.nan,
                                          'Current_Profit': np.nan,
                                          'Total_Profit': np.nan,
                                          'Reward': np.nan}).iloc[0:self.window_size,-4:]
        self.tracker.insert(loc=0, column='Date', value=self.day.Date)

        # Tradebook for closed trades
        self.trades = pd.DataFrame(columns=['Opentime', 'Closetime','Openindex', 'Closeindex',
                                            'Open', 'Close', 'Profit', 'Direction']) 
        self.trade_template = {'Opentime': None, 'Closetime': None,'Openindex': None, 'Closeindex': None, 
                                'Open': None, 'Close':None, 'Profit':None, 'Direction':None}
        self.trade = self.trade_template.copy()
        
        if self.verbose > 0:
            print(f'New Day started: {self.day.Date.iloc[self.window_size]}')
        
        # Beginning state
        self.day = self.day.to_numpy()
        state = self.day[self.idx:self.idx + self.window_size, 1:]     

        return state
####################################################################################

if __name__ == "__main__":
    window_size = 100
    gen = DataGenerator(symbol="SP500_M1",
                        fraction=[2e6, 10000],
                        window_size=window_size)
    env = environment(gen, normalization=False)
  
    n = 10
    t = time.time()
    for i in range(n):
        state = env.reset()
        while True:
            state, reward, done = env.step(action=np.random.choice([0,1,2,3], 1))
            if done:
                break
    print(f'\nDays/s: {round(n/(time.time()-t), 2)}')
    print(f'\nNumber of Trades: {len(env.trades)}\nTotal Profit: {np.sum(env.trades["Profit"])}')
    
  
