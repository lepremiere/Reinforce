import sys
import logging
import numpy as np
import pandas as pd
from multiprocessing import process

from tensorflow.python.keras.backend import dtype
from libs.datagen import DataGenerator

class Environment():

    def __init__(self, in_q, out_q, news_q, market, settings) -> None:
        self.verbose = settings['verbose']
        self.window_size = settings['window_size']
        self.normalization = settings['normalization']
        self.shuffle_days = settings['shuffle_days']
        self.future = settings['future']
        self.news_q = news_q
        self.data_gen_in_q = in_q
        self.data_gen_out_q = out_q
        self.data_gen_in_q.put(self.shuffle_days)
        pdt, sample = self.data_gen_out_q.get()
        self.observation_space = [self.window_size, np.shape(sample)[1]-1] # -1 bc date is not passed
        self.action_space = [0,1,2]
        self.action_space_n = len(self.action_space)

        self.market = market
        self.tracker_list = {'Action': 0,
                            'Position': 0,
                            'Open_Price': 0,
                            'Nbars': 0,
                            'Current_Profit': 0,
                            'Total_Profit': 0,
                            'Reward': 0}
####################################################################################
    def reset(self):
        self.data_gen_in_q.put(self.shuffle_days)
        pdt, self.day = self.data_gen_out_q.get()
        self.len = len(self.day) - self.window_size

        # Normalize to Open of first data
        if self.normalization:
            self.day.loc[:, pdt] = self.day.loc[:, pdt] - self.day.open.iloc[self.window_size]
        # Params
        self.idx = 0 

        # Recording trading activity
        self.tracker = self.day.assign(**self.tracker_list).iloc[:,-len(self.tracker_list):]

        # Tradebook for closed trades
        self.trades = pd.DataFrame(columns=['Opentime', 'Closetime','Openindex', 'Closeindex', 'Duration'
                                            'Open', 'Close', 'Profit', 'Direction']) 
        self.trade_template = {'Opentime': None, 'Closetime': None,'Openindex': None, 'Closeindex': None,
                                 'Duration': None, 'Open': None, 'Close':None, 'Profit':None, 'Direction':None}
        self.trade = self.trade_template.copy()
                    
        # Starting state
        self.day = self.day.to_numpy()
        state = (np.array(self.day[self.idx+1:self.window_size + self.idx + 1 + self.future, 1:]).astype(np.float32),\
                 np.array(self.tracker.iloc[self.idx + self.window_size, 0:].to_numpy()).astype(np.float32))

        return state
####################################################################################    
    def step(self, action):
        
        # COMMENT!
        reward = 0
        # Terminal state 
        if self.idx < self.len - 2 - self.future:
            reward = 0
            profit = 0
            open_price = 0
            nbars = 0
            current_profit = 0
            position = 0
            some = 0

            if self.trade['Opentime'] == None:
                if action == 0:
                    reward = 0
                else:
                    self.open_position(action) 
                    position = self.trade['Direction']
                    current_profit = (self.day[self.idx + self.window_size, 7] - self.trade['Open']) * position
        
            else:
                open_price = self.trade['Open']
                nbars = self.idx - self.trade['Openindex']
                position = self.trade['Direction']
                current_profit = (self.day[self.idx + self.window_size, 7] - self.trade['Open']) * position
                some = current_profit 

                if action == 0 or (position == 1 and action == 2) or (position == -1 and action == 1):
                    self.close_position() 
                    profit = self.trade['Profit']
                    reward = profit
                    del self.trade
                    self.trade = self.trade_template.copy()
                elif position > 0 and action == 2:
                    reward = some
            done = False
            total_profit = self.tracker['Total_Profit'].iloc[self.idx-1] + profit
            reward += some
            # reward += -10
            
            self.tracker.iloc[self.idx, :] = [action, position, open_price, nbars, current_profit, total_profit, reward]
            # if self.verbose > 0 and self.name == '1':
            #     # self.news_q.put(('Action', [action, position, open_price, nbars, current_profit, total_profit, reward]))
            #     pass
        
        else:
            if self.trade['Opentime'] != None:
                self.close_position()
                action = 3
                open_price = self.trade['Open']
                nbars = self.idx - self.trade['Openindex']
                position = self.trade['Direction']
                current_profit = (self.day[self.idx + self.window_size, 7] - self.trade['Open']) * position
                profit = self.trade['Profit']
                reward = profit
                total_profit = self.tracker['Total_Profit'].iloc[self.idx-1] + profit
                self.tracker.iloc[self.idx, :] = [action, position, open_price, nbars, current_profit, total_profit, reward]
            else:
                total_profit = self.tracker['Total_Profit'].iloc[self.idx-1]
                self.tracker.iloc[self.idx, :] = [0, 0, 0, 0, 0, total_profit, 0]
            done = True  

        if len(self.trades) < 1:
            self.trades.append([0,0,0,0,0,0,0,0,0])      
        
        self.idx += 1
        next_state = (np.array(self.day[self.idx+1:self.window_size + self.idx + 1 + self.future, 1:]).astype(np.float32),\
                       np.array(self.tracker.iloc[self.idx + self.window_size, 0:].to_numpy()).astype(np.float32))

        return next_state, reward, done

####################################################################################   
    def open_position(self, action):
        # 7 = Close in df. 4 = Open price in tradebook
        del self.trade
        self.trade = self.trade_template.copy()
        direction = action == 1
        self.trade['Opentime'] = self.day[self.window_size + self.idx, 0]
        self.trade['Openindex'] = self.idx
        self.trade['Open'] = self.day[self.window_size + self.idx, 7] + (self.market['Spread'] * (-1 + 2*direction))
        self.trade['Direction'] = (-1 + 2*direction)

####################################################################################    
    def close_position(self):
        self.trade['Closetime'] = self.day[self.window_size + self.idx, 0]
        self.trade['Closeindex'] = self.idx
        self.trade['Duration'] = self.trade['Closeindex'] - self.trade['Openindex']
        self.trade['Close'] = self.day[self.window_size + self.idx, 7]
        self.trade['Profit'] = ((self.trade['Close'] -\
                                self.trade['Open']) *\
                                self.trade['Direction']) 
        self.trades = self.trades.append(self.trade, ignore_index=True) 
        
        if self.verbose > 0:
            self.news_q.put(('Trade', self.trade))    
        
####################################################################################

if __name__ == "__main__":
    pass

    
  
