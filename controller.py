import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

from libs.environment import Environment
from libs.datagen import DataGenerator
from libs.buffer import ReplayBuffer
from libs.agent import Agent
from libs.worker import Worker


class Controller:
    def __init__(self, env, agent, worker) -> None:
        pass


if __name__ == "__main__":

    BATCH_SIZE = 64
    NUM_EPISODES = 5
    WINDOW_SIZE = 5
    NORMALIZATION = False
    VERBOSE = 1

    settings = {'batch_size': BATCH_SIZE,
                'num_episodes': NUM_EPISODES,
                'window_size': WINDOW_SIZE,
                'normalization': NORMALIZATION,
                'verbose': VERBOSE}

    gen = DataGenerator(symbol="SP500_M1",
                        fraction=[1, 1e4],
                        window_size=WINDOW_SIZE)
    buffer = ReplayBuffer(buffer_size=int(1e5), batch_size=BATCH_SIZE)
    env = Environment(DataGen=gen, normalization=NORMALIZATION, verbose=VERBOSE)
    agent = Agent(env)

    num_workers = 6
    agent_in_q = mp.JoinableQueue()
    workers = [Worker(name=str(i),
                      gen=gen,
                      task_queue=agent_in_q,
                      settings=settings) \
                for i in range(num_workers)]

    for w in workers:
        w.daemon = True
        w.start() 
    
    for i in range(100):
        agent_in_q.put(i)

    for i in range(num_workers):
        agent_in_q.put(None)

    agent_in_q.join()
    for w in workers:
        w.join()