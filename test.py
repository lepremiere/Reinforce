import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import numpy as np
from tqdm import tqdm
from environment import environment
from libs.datagen import DataGenerator
# from agent import Agent


gen = DataGenerator(symbol="SP500_M1",
                    fraction=[0, 1e4],
                    window_size=100)

env = environment(DataGen=gen, normalization=False, verbose=1)
# print(f'\nAction space: \n{env.action_space}')
# print(f'\nObservation space: \n{env.observation_space}')

state = env.reset()
# print(f'\nInitial State: \n{state}')

for _ in range(10):
    env.reset()
    while True:
        new_state, reward, done = env.step(action=np.random.choice([0,1,2,3], 1))
        if done:
            break


