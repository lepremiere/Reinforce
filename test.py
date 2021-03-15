# import time 
# from backtest import *

# if __name__ == "__main__":

#     agent = Agent()
#     bt = Backtesting(symbol='SP500',
#                      timeframe='M1',
#                      agent=agent,
#                      fraction=(2000000, 100000),
#                      equity=10000,
#                      risk=0.1)
#     bt.reset()
#     bt.run()
#     bt.plot(dates=True, week_lines=True, plot_trades=True)
   
from environment import make_environment
import pandas as pd 
import numpy as np 
from itertools import compress
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from lib.tech_indicators import get_technical_indicators

df = pd.read_csv('D:/Data/SP500_M1.csv', nrows=100000)
df.Date = pd.to_datetime(df.Date)
days = [g  for n, g in df.groupby(pd.Grouper(key='Date', freq='D'))]
idx = [not day.empty for day in days]
days = list(compress(days, idx))

X_train, X_test = train_test_split(days, test_size=0.2, random_state=0)
print(f'Days for training: {len(X_train)}, Days for testing: {len(X_test)}')

env = make_environment(X_train[0])
