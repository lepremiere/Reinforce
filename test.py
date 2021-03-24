import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from libs.environment import environment
from libs.datagen import DataGenerator
from libs.buffer import ReplayBuffer
from libs.agent import Agent

if __name__ == "__main__":
    batch_size = 64
    num_episodes = 5
    window_size = 5

    while True:
        gen = DataGenerator(symbol="SP500_M1",
                            fraction=[1, 1e4],
                            window_size=window_size)
        buffer = ReplayBuffer(buffer_size=int(1e5), batch_size=batch_size)
        env = environment(DataGen=gen, normalization=False, verbose=4)
        agent = Agent(env)

        total_rewards = []
        trades = []
        t = time.time()
        plt.ion()
        f, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax1.set_title('Profit')
        ax2.set_title('Reward')
        profit = []
        for episode in range(num_episodes):
            total_reward = 0.0
            state = env.reset()
            dim = np.shape(env.day)[1]-1   
            state[0] = np.reshape(np.array(state[0]), (1, window_size, dim))

            while True:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action, agent.epsilon)
                next_state[0] = np.reshape(next_state[0], (1, window_size, dim))

                advantage, value = agent.get_values(state, action, reward, next_state, done)
                buffer.add(state, advantage, value, [reward])

                total_reward += reward
                state = next_state

                if done:
                    profit.append(np.sum(env.trades['Profit']))
                    trades.append(env.trades)
                    total_rewards.append(total_reward)
                    mean_total_rewards = np.mean(total_rewards[-min(len(total_rewards), 10):])

                    ax1.clear()
                    ax1.plot(total_rewards, color='g')
                    ax2.clear()
                    ax2.plot(profit, color='b')
                    ax1.axhline(0, color='k')
                    ax2.axhline(0, color='k')
                    plt.pause(0.05)
                    plt.savefig('./storage/performance.png')

                    print("Episode: ", episode+1,
                        " Total Reward: ", total_reward,
                        " Mean: ", mean_total_rewards,
                        " Time: ", round(time.time()-t,2), " s")
                    break

            if buffer.__len__() > batch_size:
                for _ in range(5):
                    samples = buffer.sample(skewed=True)
                    print(np.shape(samples), samples[0])
   

        plt.show()   
        # trades = pd.concat(trades)
        # long = np.array(trades['Direction'] > 0 )
        # won = np.array(trades['Profit'] > 0)
        # short = trades['Direction'] < 0

        # print(f'\nDays/s: {round(num_episodes/(time.time()-t), 2)}')
        # print(f'\nNumber of Trades: {len(trades)}\nTotal Profit: {np.round(np.sum(trades["Profit"]),2)} €',
        #         f'\nAverage Return: {np.round(np.mean(trades["Profit"]),2)} ({np.round(np.min(trades["Profit"]),2)}, {np.max(trades["Profit"])})',
        #         f'\nWon Long: {np.round(np.sum(np.sum([long, won], axis=0) == 2)/np.sum(long)*100,2)} %',
        #         f'\nWon Short: {np.round(np.sum(np.sum([short, won], axis=0) == 2)/np.sum(long)*100,2)} %')
        # break


