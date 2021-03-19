import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import tensorflow as tf
from environment import environment
from libs.datagen import DataGenerator
from libs.buffer import ReplayBuffer
from agent import Agent

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

batch_size = 10000
num_episodes = 1000
window_size = 100

while True:
    gen = DataGenerator(symbol="SP500_M1",
                        fraction=[1e6, 1e6],
                        window_size=window_size)
    buffer = ReplayBuffer(buffer_size=int(1e6), batch_size=batch_size)
    env = environment(DataGen=gen, normalization=True, verbose=2)
    agent = Agent(env)

    total_rewards = []
    trades = []
    t = time.time()
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot()
    profit = []
    for episode in range(num_episodes):
        total_reward = 0.0
        state, game = env.reset()
        dim = np.shape(env.day)[1]-1   
        state = np.reshape(np.array(state), (1, window_size, dim))

        while True:
            action = agent.get_action(state, game)
            next_state, game, reward, done = env.step(action)
            next_state = np.reshape(next_state, (1, window_size, dim))

            advantage, value = agent.get_values(state, game, action, reward, next_state, done)
            buffer.add(state, advantage, value, game)

            total_reward += reward
            state = next_state

            if done:
                profit.append(np.sum(env.trades['Profit']))
                trades.append(env.trades)
                total_rewards.append(total_reward)
                mean_total_rewards = np.mean(total_rewards[-min(len(total_rewards), 10):])
                ax.clear()
                ax.plot(total_rewards)
                ax.plot(profit)
                fig.canvas.draw()
                fig.canvas.flush_events()

                print("Episode: ", episode+1,
                    " Total Reward: ", total_reward,
                    " Mean: ", mean_total_rewards,
                    " Time: ", round(time.time()-t,2), " s")
                break
        if buffer.__len__() > batch_size:
            samples = buffer.sample()
            states = np.array([sample.state[0] for sample in samples])
            advantages = np.array([sample.advantage[0] for sample in samples])
            values = np.array([sample.value[0] for sample in samples])
            games = np.array([sample.game[0] for sample in samples])

            agent.update_policy(states, games, advantages, values)
        
    trades = pd.concat(trades)
    long = np.array(trades['Direction'] > 0 )
    won = np.array(trades['Profit'] > 0)
    short = trades['Direction'] < 0

    print(f'\nDays/s: {round(num_episodes/(time.time()-t), 2)}')
    print(f'\nNumber of Trades: {len(trades)}\nTotal Profit: {np.round(np.sum(trades["Profit"]),2)} €',
            f'\nAverage Return: {np.round(np.mean(trades["Profit"]),2)} ({np.round(np.min(trades["Profit"]),2)}, {np.max(trades["Profit"])})',
            f'\nWon Long: {np.round(np.sum(np.sum([long, won], axis=0) == 2)/np.sum(long)*100,2)} %',
            f'\nWon Short: {np.round(np.sum(np.sum([short, won], axis=0) == 2)/np.sum(long)*100,2)} %')
    break


