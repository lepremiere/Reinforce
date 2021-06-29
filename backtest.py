import time
t = time.time()
import numpy as np 
from pandas import read_csv, to_datetime, DataFrame, Grouper, Timedelta, concat
from matplotlib.pyplot import subplots, show
import mplfinance as mpf
import tqdm
from multiprocessing import Pool, cpu_count, freeze_support
from multiprocessing.pool import ThreadPool
from mplfinance._utils import IntegerIndexDateTimeFormatter

from libs.agent import *

####################################################################################################
# Class
class Backtesting():
    def __init__(self, symbol, timeframe, agent, fraction=(0, 0), equity=100, risk=0.1):
        self.symbol = symbol
        self.timeframe = timeframe
        new_file = 'D:/Data/' + symbol + '_' +  timeframe + '.csv'

        self.data = read_csv(new_file,
                            header=0,
                            parse_dates=False,
                            skiprows=range(1,fraction[0]+1),
                            nrows=fraction[1])

        # Generating Dataframes
        time = to_datetime(self.data['Date'],
                            format='%Y%m%d %H:%M:%S',
                            errors='coerce')
        self.data = self.data.iloc[:, 2:]
        self.data['Date'] = time   
        self.data, n = get_technical_indicators(self.data)
        self.data['Index'] = self.data.index - n

        # Daily grouping
        self.days = [g  for n, g in self.data.groupby(Grouper(key='Date', freq='D'))]
        self.data = self.data.set_index(time.iloc[n:])
        self.l = len(self.data)
        self.trades = DataFrame( columns=['Opentime', 'Closetime','Openindex', 'Closeindex',
                                          'Open', 'Close', 'Profit', 'Volume', 'Direction'])

        # Properties                                    
        self.market = {
                        'Spread': 1.2,
                        'MinLot': 0.1,
                        'Leverage': 20,
                        'MarginCall': 0.75
                        }
        self.settings = {
                        'Risk': risk,
                        'MaxDrawDown': 0.9,
                        'TradingPause': 0
                        }  

        self.equity_start = equity
        self.equity = equity
        self.agent = agent
        self.profits = []
        self.actions = []  
        print('Initialized.')

    
    def open(self, price, time, index, direction):
        return [str(time), index, price, direction]

    def close(self, price, positions, time, index):
        # positions[0] = time, index, price, direction  
        profit = (price - positions[0][2]) * positions[0][3]
        trade = [to_datetime(positions[0][0]),
                to_datetime(time),
                positions[0][1],
                index,
                positions[0][2],
                price,
                profit,
                positions[0][3]]
        return profit, trade

    def check_market_conditions(self, trades, time, n=5):
        time = to_datetime(time)
        if  (time.hour < 14 and time.minute < 30) or (time.hour > 21): # Day over
                return False, 3
        # elif len(trades) > 0 \
        #         and time - to_datetime(trades['Closetime'][-1]) < Timedelta(minutes=n): 
        #         return False, 0
        else: 
                return True, 0

    def next(self, data):
        actions = []
        positions = []
        trades = DataFrame(columns=['Opentime', 'Closetime','Openindex', 'Closeindex',
                                        'Open', 'Close', 'Profit', 'Direction'])
        profits = []
        size_data = len(data)

        for i in range(size_data):
            price = data['Close'].iloc[i]
            time = data['Date'].iloc[i]
            index = data['Index'].iloc[i]        

            # Decision Making of the Agent
            check, check_action = self.check_market_conditions(trades, time, 5)
            if i > 15:
                action = self.agent.get_action(data.iloc[i-2:i])
            else:
                action = check_action
            if i+15 > size_data:
                action = 3
          
            # Action Execution
            if action == 0: pass
            if len(positions) > 0:
                status = positions[0][-1] + action
                if action == 3 or status == 0: 
                    profit, trade = self.close(price, positions, time, index)
                    profits.append(profit)
                    trades.loc[trade[0]] = trade
                    positions = []
                if action == 1:  positions.append(self.open(price, time, index, 1))  
                if action == 2: positions.append(self.open(price, time, index, -1))
            else:
                if action == 1:  positions.append(self.open(price, time, index, 1))  
                if action == 2: positions.append(self.open(price, time, index, -1))

            actions.append(action)

        return actions, trades, profits

    def run(self):
        n = 6
        t1 = time.time()
        results = []
        p = Pool(n)

        # Do stuff in parallel
        results = p.map(self.next, self.days)
        for actions, trades, profits in results:
            self.actions = np.concatenate([self.actions, actions])
            self.trades = concat([self.trades, trades])
            self.profits = np.concatenate([self.profits, profits]) 

        # Calculate volumes and profits
        volumes = []
        profits = []
        equities = []
        equity = self.equity_start
        for j in range(len(self.trades)):
            if self.settings['Risk'] == 0:
                volume = 1
            else:
                volume = (equity * self.settings['Risk']) / (self.trades['Open'].iloc[j] / self.market['Leverage'])
            if volume < self.market['MinLot']:
                profit = 0
                volume = 0
                profit = 0
            else:
                profit = (self.trades['Profit'].iloc[j] - self.market['Spread']) * volume
            equity += profit
            equities.append(equity)
            volumes.append(volume)
        
        self.trades['Volume'] = volumes
        self.trades['Equities'] = equities
        t2 = time.time() 

        print(str(len(self.days)), 'Days @',
              str(round(len(self.days)/(t2-t1), 1)), 'Days/s with',
              n, 'Worker in', str(round(t2-t1, 1)), 'seconds.',
              'End Result:', str(round(equity)))
        
    def reset(self):
        self.profits = []
        self.actions = []
        self.trades = DataFrame( columns=['Opentime', 'Closetime','Openindex', 'Closeindex',
                                             'Open', 'Close', 'Profit', 'Direction'])

####################################################################################################
####################################################################################################
####################################################################################################
# Plotting
    def plot(self, dates=True, week_lines=True, plot_trades=True):
        t = time.time()
        fig, axs = subplots(nrows=4, ncols=1, sharex=True, gridspec_kw={'height_ratios':[5,3,1,2]})
        
        # ax = mpf.plot(self.data,
        #          type='candle',
        #          style='yahoo',
        #          mav=(9,21),
        #          ax=axs[0],
        #          axtitle=self.symbol + '_' + self.timeframe,
        #          returnfig=True)

        # Time parameter
        if dates:
            x = to_datetime(self.data.index)
            time_fmt = ['Opentime', 'Closetime']
        else:
            x = range(self.l)
            time_fmt = ['Openindex', 'Closeindex']

        axs[0].plot(x, self.data['upper_band'],     color='lightsteelblue')
        axs[0].plot(x, self.data['lower_band'],     color='lightsteelblue')        
        axs[0].plot(x, self.data['ma21'],           color='orange')
        axs[0].plot(x, self.data['ma9'],            color='gold')
        axs[0].plot(x, self.data['Close'].values,   color='blue')

        axs[0].set_title('Symbol: '     + self.symbol + '      '+\
                         'Timeframe: '  + self.timeframe + ' \n' +\
                         str(len(self.days)) + ' Days: ' +\
                         str(to_datetime(self.days[0]['Date'].iloc[0]).date()) + ' --- ' +\
                         str(to_datetime(self.days[-1]['Date'].iloc[-1]).date()) )
        axs[0].set_ylabel('Close [$]')      

        # Plotting Profit
        line1, = axs[1].plot(self.trades[time_fmt[1]],
                             self.trades['Equities']) 
        line2, = axs[1].plot(self.trades[time_fmt[1]], 
                             self.equity_start + np.cumsum(self.trades['Profit'].values))
        line2.set_drawstyle("steps-post")
        line1.set_drawstyle("steps-post")
        ratio = np.sum(np.array(self.trades['Profit'].values) > 0, axis=0) / len(self.trades) * 100
        axs[1].set_title('Profit: ' + str(round(np.sum(self.trades['Profit'].values),1)) +
                         '$ Total Trades: ' + str(len(self.trades)) +
                         ' Win Ratio: ' + str(round(ratio, 2)) + '%')    
        axs[1].set_ylabel('Cummulative Profit [$]')
        xmin, xmax = axs[1].get_xlim()
        axs[1].hlines(self.equity_start, xmin=xmin, xmax=xmax, color='red', alpha=0.5)
        axs[1].legend((line1,line2), ('Dynamic', 'Static'), loc='upper left')

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
        if len(self.trades) > 1000:
            filt = 0
        else:
            filt = 0
        
        if plot_trades:
            X, Y = self.trades[time_fmt], self.trades[['Open', 'Close']] 

            for i in range(len(X)):
                if self.trades['Direction'].iloc[i] == 1:
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
        if week_lines:
            weekly_lines = []
            daily_lines = []
            for i in range(1, self.l): 
                w1 = to_datetime(self.data.index[i]).week
                w2 = to_datetime(self.data.index[i-1]).week
                d1 = to_datetime(self.data.index[i]).day
                d2 = to_datetime(self.data.index[i-1]).day
                if w2 < w1:
                    if dates:
                        weekly_lines.append(self.data.index[i])
                    else:
                        weekly_lines.append(i)
                if d2 < d1:
                    if dates:
                        daily_lines.append(self.data.index[i])
                    else:
                        daily_lines.append(i)

            ymin, ymax = axs[0].get_ylim()
            axs[0].vlines(weekly_lines,ymin=ymin,ymax=ymax)
            axs[0].vlines(daily_lines,ymin=ymin,ymax=ymax, alpha=0.3)

        print('Plot: ', str(round(time.time()-t,1)))
        show()

if __name__ == "__main__":
    
    agent = Agent()
    bt = Backtesting(symbol='US30',
                     timeframe='H1',
                     agent=agent,
                     fraction=(10000, 10000),
                     equity=1000,
                     risk=0.0)
    bt.reset()
    bt.run()
    bt.plot(dates=True, week_lines=True, plot_trades=True)

