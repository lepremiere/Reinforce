import time 
from backtest import *

if __name__ == "__main__":

    agent = Agent()
    bt = Backtesting(symbol='US30',
                     timeframe='M1',
                     agent=agent,
                     fraction=(0, 0),
                     equity=10000,
                     risk=0.1)
    bt.reset()
    bt.run()
    bt.plot(dates=False, week_lines=True, plot_trades=True)
   