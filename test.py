import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from environment import environment
from lib.tools import get_data
from agent import Agent


X_train, X_test = get_data('D:/Data/SP500_M1.csv', fraction=[1e6,1e5], test_size=0.2)
print(f'Days for training: {len(X_train)}, Days for testing: {len(X_test)}')

env = environment(X_train)
print(f'\nAction space: \n{env.action_space()}')
print(f'\nObservation space: \n{env.observation_space()}')
# input('?')

agent = Agent(env)
state = env.reset()
print(f'\nInitial State: \n{state}')
# input('?')

total_rewards = agent.train(num_episodes=10)