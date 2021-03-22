import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import multiprocessing as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from libs.environment import Environment
from libs.datagen import DataGenerator
from libs.buffer import ReplayBuffer
from libs.agent import Agent
from libs.worker import Worker


class Controller:
    def __init__(self, env, agent, worker) -> None:
        pass


if __name__ == "__main__":
    t2 = time.time()

    settings = {'batch_size': 64,
                'num_episodes': 1,
                'window_size': 5,
                'normalization': False,
                'buffer_size': int(1e4),
                'verbose': 1}

    gen = DataGenerator(symbol="SP500_M1", fraction=[1, 1e3], settings=settings)
    buffer = ReplayBuffer(settings=settings)
    env = Environment(DataGen=gen, settings=settings)
    agent = Agent(env)
    
    agent_in_q = mp.JoinableQueue()
    num_workers = 1
    workers = [Worker(name=str(i),
                      gen=gen,
                      buffer=buffer,
                      task_queue=agent_in_q,
                      settings=settings) \
                for i in range(num_workers)]

    #################################################
    # Start  and terminate processes
    for w in workers:
        w.daemon = True
        w.start() 

    for i in range(1):
        agent_in_q.put(i)

    for i in range(num_workers):
        agent_in_q.put(None)
    
    agent_in_q.join()
    for w in workers:
        w.join()
    ##################################################
    
    print(time.time()-t2)
