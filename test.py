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
from lib.tools import get_data
from agent import Agent

X_train, X_test = get_data('D:/Data/SP500_M1.csv', fraction=[1e6,1e5], test_size=0.2)
print(f'Days for training: {len(X_train)}, Days for testing: {len(X_test)}')

env = make_environment(X_train[0])
print(f'\nAction space: \n{env.action_spec()}')
print(f'\nObservation space: \n{env.observation_spec()}')

state, done = env.reset()
agent = Agent(env)

done = False
while not done:
    action = agent.get_action(state)
    state, done = env.step(action)
    print(f'Observation: {state}, Action: {action}')
    


