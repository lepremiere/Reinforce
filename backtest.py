import numpy as np 
import pandas as pd
from pandas import read_csv, to_datetime, to_timedelta
import matplotlib.pyplot as plt 
import tqdm
from multiprocessing import Pool, cpu_count
import time


from tech_indicators import * 
from agent import *

class Backtesting:
    def __init__(self, symbol, timeframe, agent, fraction=(0, 0), equity=100):
        self.symbol = symbol
        self.timeframe = timeframe
        new_file = 'D:/Data/' + symbol + '_' +  timeframe + '.csv'

        if fraction[1] > 0:
            self.data = read_csv(new_file,
                                header=0,
                                parse_dates=False,
                                skiprows=range(1,fraction[0]+1),
                                nrows=fraction[1])
        elif fraction[1] == 0:
            self.data = read_csv(new_file,
                                 header=0,
                                 skiprows=range(1,fraction[0]+1),
                                 parse_dates=False)

        time = to_datetime(self.data['Date'],
                            format='%Y%m%d %H:%M:%S',
                            errors='coerce')

        self.data = self.data.iloc[:, 2:]
        self.data['Date'] = time
        self.data, n = get_technical_indicators(self.data)
        self.days = [g  for n, g in self.data.groupby(pd.Grouper(key='Date', freq='D'))]
        self.data = self.data.set_index(time.iloc[n:])

        self.l = len(self.data)

        self.trades = pd.DataFrame( columns=['Opentime', 'Closetime','Openindex', 'Closeindex',
                                             'Open', 'Close', 'Profit', 'Volume', 'Direction'])
        # self.status = {

        #                 }
        self.market = {
                        'Spread': 1.2,
                        'MinLot': 0.1,
                        'Leverage': 30,
                        'MarginCall': 0.75
                        }

        self.settings = {
                        'Risk': 0.8,
                        'MaxDrawDown': 0.9,
                        'TradingPause': 0
                        }  

        self.equity_start = equity
        self.equity = equity
        self.agent = agent
        self.profits = []
        self.actions = []
        self.counter = 0
    
    def open(self, price, volume, time, i, direction):
        if volume > self.market['MinLot']:
            return [str(time), i, price, volume, direction]
        else:
            print('Insufficient Funds!')

    def close(self, price, positions, time, index):
        open_time, trade_info = positions[0][0], positions[0][1:]     
        profit = ((price - trade_info[1]) * trade_info[3] - self.market['Spread']) * trade_info[2]
        trade = [pd.to_datetime(open_time),
                pd.to_datetime(time),
                trade_info[0],
                index,
                trade_info[1],
                price,
                profit,
                trade_info[2],
                trade_info[3]]

        return profit, trade, []
            
    def check_market_conditions(self, trades, time):
        # if self.equity/self.equity_start < (1 - self.settings['MaxDrawDown']):
        #     return False, 2
        if  (time.hour < 14 and time.minute < 30) or (time.hour > 21): # Day over
            return False, 0
        elif len(trades) > 0 \
            and time - pd.to_datetime(trades['Closetime'][-1]) < pd.Timedelta(minutes=self.settings['TradingPause']): 
            return False, 2
        else: 
            return True, 0   
    
    def next(self, data):
        # agent = Agent()
        actions = []
        positions = []
        trades = pd.DataFrame(columns=['Opentime', 'Closetime','Openindex', 'Closeindex',
                                        'Open', 'Close', 'Profit', 'Volume', 'Direction'])
        profits = []
        size_data = len(data)

        for i in range(size_data):
            price = data.iloc[i, 2]
            time = pd.to_datetime(data['Date'].iloc[i])  
            volume = round((self.equity * self.settings['Risk']) / (price / self.market['Leverage']), 2)          

            # Decision Making of the Agent
            check, check_action = self.check_market_conditions(trades, time)
            if i > 2:
                action = self.agent.get_action(data.iloc[i-2:i])
            else:
                action = check_action

            if i+3 > size_data:
                action = 2
          
            # Action Execution
            if action == 0: pass
            if len(positions) > 0:
                status = action + positions[0][-1]
                if action == 2: 
                    profit, trade, _ = self.close(price, positions, time, self.counter)
                    profits.append(profit)
                    trades.loc[trade[0]] = trade
                    positions = []
            else:
                if action == 1:  positions.append(self.open(price, volume, time, self.counter, 1))  
                if action == -1: positions.append(self.open(price, volume, time, self.counter, -1))

            actions.append(action)
            self.counter += 1

        return actions, trades, profits

    def run(self):
        n = 12
        t1 = time.time()
        results = []
        p = Pool(n)
        # for x in tqdm.tqdm(p.imap_unordered(self.next, self.days), total=len(self.days)):
        #     results.append(x)
        results = p.map(self.next, self.days)

        for actions, trades, profits in results:
            self.actions = np.concatenate([self.actions, actions])
            self.trades = pd.concat([self.trades, trades])
            self.profits = np.concatenate([self.profits, profits])  
        t2 = time.time() 
        print(str(len(self.days)), 'Days @',
              str(round(len(self.days)/(t2-t1), 1)), 'Days/s with',
              n, 'Worker in', str(round(t2-t1, 1)), 'seconds')
    
    def reset(self):
        self.profits = []
        self.actions = []
        self.trades = pd.DataFrame( columns=['Opentime', 'Closetime','Openindex', 'Closeindex',
                                             'Open', 'Close', 'Profit', 'Volume', 'Direction'])

    def plot(self, time=True):
        fig, axs = plt.subplots(nrows=4, ncols=1, gridspec_kw={'height_ratios': [5,3,1,2]}, sharex=True)

        # Time parameter
        if time:
            x = pd.to_datetime(self.data.index)
            time_fmt = ['Opentime', 'Closetime']
        else:
            x = range(self.l)
            time_fmt = ['Openindex', 'Closeindex']

        axs[0].plot(x, self.data['upper_band'], color='lightsteelblue')
        axs[0].plot(x, self.data['lower_band'], color='lightsteelblue')        
        axs[0].plot(x, self.data['ma21'], color='orange')
        axs[0].plot(x, self.data['ma7'], color='gold')
        axs[0].plot(x, self.data['Close'].values, color='blue')
        axs[0].set_title('Symbol: ' + self.symbol +\
                         '      Timeframe: ' + self.timeframe +  ' \n' +\
                         str(len(self.days)) + ' Days: ' +\
                         str(pd.to_datetime(self.days[0]['Date'].iloc[0]).date()) + ' --- ' +\
                         str(pd.to_datetime(self.days[-1]['Date'].iloc[-1]).date()) )
        axs[0].set_ylabel('Close [$]')

        # Plotting Profit
        line, = axs[1].plot(self.trades[time_fmt[1]], self.equity_start + np.cumsum(self.trades['Profit'].values)) 
        line.set_drawstyle("steps-post")
        ratio = np.sum(np.array(self.trades['Profit'].values) > 0, axis=0) / len(self.trades) * 100
        axs[1].set_title('Profit: ' + str(round(np.sum(self.trades['Profit'].values),1)) +
                         ' Total Trades: ' + str(len(self.trades)) +
                         ' Win Ratio: ' + str(round(ratio, 2)) + '%')    
        axs[1].set_ylabel('Cummulative Profit [$]')

        # Plotting Volume
        axs[2].bar(self.trades[time_fmt[1]],
                   self.trades['Volume'],
                   color='yellow',
                   width=0.0001,
                   edgecolor='black',
                   linewidth=0.5)
        axs[2].set_title('Volume') 
        axs[2].set_ylabel('Volume [Lots]')

        # Plotting Actions
        line, = axs[3].plot(x, self.actions)
        line.set_drawstyle("steps-pre") 
        axs[3].set_title('Actions')    
        axs[3].set_xlabel('Time [yyyy-mm-dd]')
        axs[3].set_ylabel('Action [a.u.]')

        # Plotting the trades
        X, Y = self.trades[time_fmt], self.trades[['Open', 'Close']] 

        filt = 0
        if len(Y) > 1e3:
            filt =  1  
        for i in range(len(X)):
            if abs(self.trades['Profit'].iloc[i]) > filt:
                if self.trades['Direction'].iloc[i] > 0:
                    marker = '^'
                else: 
                    marker = 'v'                
                if self.trades['Profit'].iloc[i] > 0:
                    color = 'green'
                elif self.trades['Profit'].iloc[i] < 0:
                    color = 'red'
                line, = axs[0].plot(X.iloc[i],Y.iloc[i], color=color)
                line.set_marker(marker)
                line.set_markeredgecolor('black')
                line.set_markerfacecolor('yellow')       

        # Daily lines for time=False
        if not time:
            lines = []
            for i in range(1, self.l): 
                t1 = pd.to_datetime(self.data.index[i]).day
                t2 = pd.to_datetime(self.data.index[i-1]).day
                if t2 < t1:
                    lines.append(i)
            ymin, ymax = axs[0].get_ylim()
            axs[0].vlines(lines,ymin=ymin,ymax=ymax)

        plt.show()


if __name__ == "__main__":

    agent = Agent()
    bt = Backtesting(symbol='US30', timeframe='M1', agent=agent, fraction=(10000, 10000), equity=10000)
    bt.reset()
    bt.run()
    bt.plot(time=True)