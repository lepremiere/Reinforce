import sys
import logging
import numpy as np
import pandas as pd
from multiprocessing import process

from tensorflow.python.keras.backend import dtype
from libs.datagen import DataGenerator

class Environment():

    def __init__(self, DataGen, settings) -> None:
        self.verbose = settings['verbose']
        self.window_size = settings['window_size']
        self.normalization = settings['normalization']
        self.datagen = DataGen
        self.observation_space = self.datagen.df.dtypes
        self.observation_space_n = [self.window_size, np.shape(self.datagen.days[0])[1]-1] # -1 bc date is not passed
        self.action_space = [0,1,2,3]
        self.action_space_n = len(self.action_space)

        self.market = {'Spread': 1.2,
                       'Risk': 0.1,
                       'Slippage': 0.5,
                       'MinLot': 0.1,
                        }
        self.tracker_list = {'Critic_Value': 0,
                            'Action': 0,
                            'Position': 0,
                            'Current_Profit': 0,
                            'Total_Profit': 0,
                            'Reward': 0}
####################################################################################
    def reset(self):
        self.day = self.datagen.get_sample(k=1)
        self.len_day = len(self.day) -self.window_size

        # Normalize to Open of first data
        if self.normalization:
            self.day.loc[:,self.datagen.price_dependent] = self.day.loc[:,self.datagen.price_dependent] -\
                                                        self.day.open.iloc[self.window_size]
        # Params
        self.len = len(self.day) - self.window_size
        self.idx = 0 

        # Recording trading activity
        self.tracker = self.day.assign(**self.tracker_list).iloc[:,-6:]

        # Tradebook for closed trades
        self.trades = pd.DataFrame(columns=['Opentime', 'Closetime','Openindex', 'Closeindex',
                                            'Open', 'Close', 'Profit', 'Direction']) 
        self.trade_template = {'Opentime': None, 'Closetime': None,'Openindex': None, 'Closeindex': None, 
                                'Open': None, 'Close':None, 'Profit':None, 'Direction':None}
        self.trade = self.trade_template.copy()
        
        if self.verbose > 0:
            print(f'New Day started: {self.day.date.iloc[self.window_size]}')
            logging.info('aaaa')
            
        # Starting state
        self.day = self.day.to_numpy()
        state = [np.asarray(self.day[self.idx:self.window_size + self.idx, 1:], dtype=object).astype(np.float32),\
                 np.asarray([self.tracker.iloc[self.idx:self.idx + self.window_size, 0:].values], dtype=object).astype(np.float32)]

        if self.verbose == 3:
                print(state[0:10, 0:10])
        return state
####################################################################################    
    def step(self, action, epsilon):
        
        # COMMENT!
        reward = 0
        # Terminal state 
        if self.idx < self.len-1:

            profit = 0
            current_profit = 0
            position = 0
            some = 0

            if self.trade['Opentime'] == None:
                if action == 0 or action == 3:
                    reward = -5
                    if action == 3:
                        reward = -10
                else:
                    self.open_position(action) 
                    position = self.trade['Direction']
                    current_profit = (self.day[self.idx + self.window_size, 7] - self.trade['Open']) * position
                    reward = -5
        
            else:
                position = self.trade['Direction']
                current_profit = (self.day[self.idx + self.window_size, 7] - self.trade['Open']) * position
                some = current_profit - (self.day[self.idx + self.window_size-1, 7] - self.trade['Open']) * position

                if action < 3:
                    if action != 0:
                        reward = -10
                else:
                    self.close_position() 
                    profit = self.trade['Profit']
                    reward = profit
                    self.trade = self.trade_template.copy()
            
            done = False
            total_profit = self.tracker['Total_Profit'].iloc[self.idx-1] + profit
            reward += some*10 
            self.tracker.iloc[self.idx, :] = [0, action, position, current_profit, total_profit, reward]
            if self.verbose == 4:
                print(f'Close: {round(self.day[self.idx + self.window_size, 7],2): >7} Action: {action: >2} Position:{position: >5}',
                      f' Current Profit: {round(current_profit,2): >7} Total Profit: {round(total_profit,2): >7} Reward: {round(reward,2): >8}',
                      f' Epsilon: {round(epsilon,3)}') 
        
        else:
            if self.trade['Opentime'] != None:
                self.close_position()
            done = True        
        
        self.idx += 1
        next_state = [np.asarray(self.day[self.idx:self.window_size + self.idx, 1:], dtype=object).astype(np.float32),\
                     np.asarray([self.tracker.iloc[self.idx:self.idx + self.window_size, 0:].values], dtype=object).astype(np.float32)]

        return next_state, reward, done

####################################################################################   
    def open_position(self, action):
        # 7 = Close in df. 4 = Open price in tradebook
        direction = action == 1
        self.trade['Opentime'] = self.day[self.window_size + self.idx, 0]
        self.trade['Openindex'] = self.idx
        self.trade['Open'] = self.day[self.window_size + self.idx, 7] + (self.market['Spread'] * (-1 + 2*direction))
        self.trade['Direction'] = (-1 + 2*direction)

####################################################################################    
    def close_position(self):
        self.trade['Closetime'] = self.day[self.window_size + self.idx, 0]
        self.trade['Closeindex'] = self.idx
        self.trade['Close'] = self.day[self.window_size + self.idx, 7]
        self.trade['Profit'] = ((self.trade['Close'] -\
                                self.trade['Open']) *\
                                self.trade['Direction']) 
        self.trades = self.trades.append(self.trade, ignore_index=True) 

        if self.verbose == 2:   
            status = self.trade["Profit"] > 0
            if status:
                outcome = 'Won'
            else:
                outcome = 'Loss'
            print(f'Opentime: {self.trade["Opentime"]}  Duration: {self.trade["Closeindex"]-self.trade["Openindex"]: >3} m  ', 
                    f'Open: {np.round(self.trade["Open"],2): >8}  Close: {np.round(self.trade["Close"],2): >8} ',
                    f'Profit: {np.round(self.trade["Profit"],2): >5}  Direction: {self.trade["Direction"]: >3} Status: {outcome: >3}')
        
####################################################################################

if __name__ == "__main__":

    window_size = 100
    gen = DataGenerator(symbol="SP500_M1",
                        fraction=[0, 1e4],
                        window_size=window_size)
    env = Environment(gen, normalization=False, verbose=1)
  
    n = 3

    for i in range(n):
        state = env.reset()
        while True:
            state, reward, done = env.step(action=np.random.choice([0,1,2,3], 1, p=[0.5, 0.1,0.1, 0.3]), epsilon=0)
            if done:
                break

    
  
