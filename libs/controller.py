import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Value, Array, Process

import multiprocessing as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from libs.environment import Environment
from libs.datagen import DataGenerator
from libs.buffer import ReplayBuffer
from libs.agent import Agent
from libs.worker import Worker
from libs.communication import *


class Controller(Process):
    def __init__(self, news_q, val, market, settings) -> None:
        Process.__init__(self)
        self.t = time.time()
        self.settings = settings
        self.num_workers = settings['num_workers']
        self.verbose = settings['verbose']
        self.market = market
        self.val = val

        # Queues
        self.task_q = mp.JoinableQueue()
        self.agent_in_q = mp.JoinableQueue()
        self.data_gen_in_q = mp.JoinableQueue()
        self.data_gen_out_q = mp.JoinableQueue()
        self.buffer_in_q = mp.JoinableQueue()
        self.buffer_out_q = mp.JoinableQueue()
        self.batch_gen_in_q = mp.JoinableQueue()
        self.distributor_in_q = mp.JoinableQueue()
        self.pipes = [mp.JoinableQueue() for _ in range(self.num_workers)]
        self.news_q = news_q

        # Processes
        self.gen =          DataGenerator(  in_q=self.data_gen_in_q,    out_q=self.data_gen_out_q,      news_q=self.news_q, val=self.val, settings=self.settings).start()
        self.batch_gen =    BatchGenerator( in_q=self.batch_gen_in_q,   out_q=self.agent_in_q,          news_q=self.news_q, val=self.val, settings=self.settings)
        self.distributor =  Distributor(    in_q=self.distributor_in_q, pipes=self.pipes,               news_q=self.news_q, val=self.val, settings=self.settings)
        self.buffer =       ReplayBuffer(   in_q=self.buffer_in_q,      out_q=self.buffer_out_q,        news_q=self.news_q, val=self.val, settings=self.settings).start()
        self.env =          Environment(    in_q=self.data_gen_in_q,    out_q=self.data_gen_out_q,      news_q=self.news_q, market=self.market, settings=self.settings)
        self.agent =    Agent(env=self.env, in_q=self.agent_in_q,       out_q=self.distributor_in_q,    news_q=self.news_q, val=self.val, settings=self.settings)
        self.workers = [Worker(name=str(i),
                        data_gen_in_q=self.data_gen_in_q,
                        data_gen_out_q=self.data_gen_out_q,
                        buffer_in_q=self.buffer_in_q,
                        buffer_out_q=self.buffer_out_q,
                        task_q=self.task_q,
                        agent_in_q=self.agent_in_q,
                        batch_gen_in_q=self.batch_gen_in_q,
                        pipe=self.pipes[i],
                        replay_buffer=self.buffer,
                        news_q=self.news_q,
                        market=self.market,
                        val=self.val,
                        settings=self.settings) \
                    for i in range(self.num_workers)]

        #################################################
        # Start and terminate processes
        self.agent.daemon = True
        self.distributor.daemon = True
        self.batch_gen.daemon = True
        self.agent.start()
        self.distributor.start()
        self.batch_gen.start() 

        for w in self.workers:
            w.daemon = True
            w.start() 
        
        t2 = time.time() 
        if self.verbose > 0:
            print('Time to start sampling: ', round(t2-self.t, 2), ' s,  Num workers: ', self.num_workers)

    def work(self, schedule):
        print('Num cycles: ', schedule[0], ', Selfplay episodes: ', schedule[1], ' + ',self.num_workers, \
              ', Training episodes: ', schedule[2])

        [self.task_q.put('play') for _ in range(self.num_workers)]
        for _ in range(schedule[0]):
            for _ in range(schedule[1]):
                self.task_q.put('play')
            for _ in range(schedule[2]):
                self.task_q.put('train')  
        self.first_run = False  

    def deinit(self):
        for w in self.workers:
            self.task_q.put(None)
        for w in self.workers:
            w.join()
        self.batch_gen.join()
        self.distributor.join()

        self.agent_in_q.put((None, None, None, None))
        self.agent_in_q.close()
        self.task_q.close()
        self.batch_gen_in_q.close()
        self.distributor_in_q.close()
        self.data_gen_out_q.close()
        self.batch_gen_in_q.close()

        for pipe in self.pipes:
            pipe.close()

if __name__ == "__main__":
    pass

