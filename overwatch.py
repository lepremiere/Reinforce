import time
import sys
import numpy as np 
import matplotlib.pyplot as plt
from multiprocessing import  Process, Pipe, JoinableQueue
from pyqtgraph.Qt import QtGui
from libs.overwatch_fig import display

class Overwatch(Process):
    def __init__(self, in_q, val, settings) -> None:
        Process.__init__(self)
        self.news_in_q = in_q
        self.val = val
        self.settings = settings
        self.counter = [0,0,0]
        self.samples = 0
        self.verbose = settings['verbose']

        self.plot_qs = [JoinableQueue() for _ in range(3)]
        p = Process(target=display, args=('bob',self.plot_qs))
        p.start()

    def run(self):
        while True:
            try:
                message, content = self.news_in_q.get(timeout=0.0)
                self.news_in_q.task_done()

                # Episode end
                if message == 'Episode_end':
                    
                    self.counter[0] += 1
                    self.samples += content[0]
                    self.plot_qs[0].put([self.counter[0], content[1]]) 

                    if self.verbose > 0:
                        print(f'Episode: {self.counter[0]: >3},  Length: {content[0]: >4}, Reward: {content[1]: >6} ({content[2]: >6},  {content[3]: >6})',
                                f'{0}')
  
                # Trades
                if message == 'Trade':
                    trade = content
                    self.plot_qs[1].put([self.counter[1], np.round(trade["Profit"],2)])
                    self.counter[1] += 1

                    if False:
                        status = trade["Profit"] > 0
                        if status:
                            outcome = 'Won'
                        else:
                            outcome = 'Loss'
                        print(f'Opentime: {trade["Opentime"]}  Duration: {trade["Closeindex"]-trade["Openindex"]: >3} m  ', 
                                f'Open: {np.round(trade["Open"],2): >8}  Close: {np.round(trade["Close"],2): >8} ',
                                f'Profit: {np.round(trade["Profit"],2): >5}  Direction: {trade["Direction"]: >3} Status: {outcome: >3}')        
                
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