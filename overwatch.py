import time
import sys
import numpy as np 
import matplotlib.pyplot as plt
from multiprocessing import  Process, Pipe, JoinableQueue
from pyqtgraph.Qt import QtGui
from libs.overwatch_fig import display

class Overwatch(Process):
    def __init__(self, in_q, val, settings, schedule) -> None:
        Process.__init__(self)
        self.news_in_q = in_q
        self.val = val
        self.settings = settings
        self.counter = [0,0,0,0,0]
        self.samples = 0
        self.verbose = settings['verbose']

        self.plot_qs = [JoinableQueue() for _ in range(7)]
        p = Process(target=display, args=('bob',self.plot_qs, settings, schedule))
        p.start()

    def run(self):
        while True:
            try:
                message, content = self.news_in_q.get(timeout=0.0)
                self.news_in_q.task_done()

                # Episode end
                if message == 'Episode_end':
                    length, mean_reward, plus, minus, mean_return, mean_duration, num_trades = content
                    self.samples += length
                    self.plot_qs[0].put([self.counter[0], mean_reward]) 
                    self.plot_qs[1].put([self.counter[0], (mean_return/length)*60*24])
                    self.plot_qs[2].put([self.counter[0], mean_duration])
                    self.plot_qs[3].put([self.counter[0], (num_trades/length)*60])
                    self.counter[0] += 1

                    if self.verbose > 0:
                        print(f'Episode: {self.counter[0]: >3},  Length: {content[0]: >4}',
                              f'Mean Reward: {mean_reward: >8} ({plus: >9},  {minus: >9}),',
                              f'Return: {round((mean_return/length)*60*24, 2): >9}/day,'
                              f'Mean Duration: {mean_duration: >6}, Num Trades: {round((num_trades/length)*60, 2): >3}/hour')

                # Epsilon
                if message == 'Epsilon':
                    epsilon = content
                    self.plot_qs[4].put([self.counter[1], epsilon])
                    self.counter[1] += 1

                # Trades
                if message == 'Trade':
                    trade = content
                    self.plot_qs[5].put([self.counter[2], np.round(trade["Profit"],2)])
                    self.counter[2] += 1     
                
                # Loss
                if message == 'Loss':
                    loss = content
                    self.plot_qs[6].put([self.counter[3], loss[0], loss[1]])
                    self.counter[3] += 1
                    print('aaa', loss)

                # Movement
                if message == 'Action':
                    pass
                    # _, action, position, current_profit, total_profit, reward = content
                    # self.plot_qs[2].put([self.counter[2], action])
                    # self.counter[2] += 1
                    # print(f'Close: {round(self.day[self.idx + self.window_size, 7],2): >7} Action: {action: >2} Position:{position: >5}',
                    #     f' Current Profit: {round(current_profit,2): >7} Total Profit: {round(total_profit,2): >7} Reward: {round(reward,2): >8}',
                    #     f' Epsilon: {round(epsilon,3)}') 
            
            except:
                if np.sum(self.val[:]) == 0:
                    break
           
        print('Overwatch is done!')
        time.sleep(1)
        print('\nTotal time elapsed: ', round(time.time()-self.settings['start_time'], 2), ' s, Speed: ',
                    round(self.samples/(time.time()-self.settings['start_time'])), ' iters/s')



if __name__ == "__main__":
    pass