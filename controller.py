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
    BATCH_SIZE = 64
    NUM_EPISODES = 5
    WINDOW_SIZE = 5
    NORMALIZATION = False
    VERBOSE = 4

    settings = {'batch_size': BATCH_SIZE,
                'num_episodes': NUM_EPISODES,
                'window_size': WINDOW_SIZE,
                'normalization': NORMALIZATION,
                'verbose': VERBOSE}

    gen = DataGenerator(symbol="SP500_M1",
                        fraction=[1, 1e5],
                        window_size=WINDOW_SIZE)
    buffer = ReplayBuffer(buffer_size=int(1e5), batch_size=BATCH_SIZE)
    env = Environment(DataGen=gen,
                      normalization=NORMALIZATION,
                      verbose=VERBOSE)
    agent = Agent(env)
    
    agent_in_q = mp.JoinableQueue()
    num_workers = 6
    workers = [Worker(name=str(i),
                      gen=gen,
                      task_queue=agent_in_q,
                      settings=settings) \
                for i in range(num_workers)]

    #################################################
    # Start  and terminate processes
    for w in workers:
        w.daemon = True
        w.start() 

    for i in range(50):
        agent_in_q.put(i)

    for i in range(num_workers):
        agent_in_q.put(None)
    
    agent_in_q.join()
    for w in workers:
        w.join()
    ##################################################
    
    print(time.time()-t2)
