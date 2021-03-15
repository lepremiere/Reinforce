from environment import make_environment
from lib.tools import get_data
from agent import Agent

X_train, X_test = get_data('D:/Data/SP500_M1.csv', fraction=[1e6,1e5], test_size=0.2)
print(f'Days for training: {len(X_train)}, Days for testing: {len(X_test)}')

env = make_environment(X_train[0])
print(f'\nAction space: \n{env.action_spec()}')
print(f'\nObservation space: \n{env.observation_spec()}')

state, reward, done = env.reset()
agent = Agent(env)

done = False
while not done:
    action = agent.get_action(state)
    state, reward, done = env.step(action)
    print(f'Observation: {state} Action: {action} Reward: {reward}')

