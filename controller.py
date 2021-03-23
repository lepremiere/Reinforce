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
from libs.communication import *


class Controller:
    def __init__(self, env, agent, worker) -> None:
        pass


if __name__ == "__main__":
    t2 = time.time()

    settings = {'batch_size': 64,
                'num_episodes': 1,
                'window_size': 10,
                'normalization': False,
                'buffer_size': int(1e4),
                'verbose': 0}
    num_workers = 1

    # Queues
    task_q = mp.JoinableQueue()
    agent_in_q = mp.JoinableQueue()
    trainer_in_q = mp.JoinableQueue()
    batch_gen_in_q = mp.JoinableQueue()
    distributor_in_q = mp.JoinableQueue()
    result_q = mp.JoinableQueue()
    pipes = [mp.JoinableQueue() for _ in range(num_workers)]

    # Processes
    gen = DataGenerator(symbol="SP500_M1", fraction=[1, 1e4], settings=settings)
    batch_gen = BatchGenerator(in_q=batch_gen_in_q, out_q=agent_in_q, k=num_workers, settings=settings).start()
    distributor = Distributor(in_q=distributor_in_q, pipes=pipes, settings=settings).start()
    buffer = 1
    env = Environment(DataGen=gen, settings=settings)
    agent = Agent(env=env, in_q=agent_in_q, out_q=distributor_in_q, settings=settings)
    workers = [Worker(name=str(i),
                      gen=gen,
                      task_q=task_q,
                      agent_in_q=agent_in_q,
                      batch_gen_in_q=batch_gen_in_q,
                      pipe=pipes[i],
                      replay_buffer=buffer,
                      result_q=result_q,
                      settings=settings) \
                for i in range(num_workers)]
    t3 = time.time() 
    print('\nTime to initiation: ',t3-t2)
    #################################################
    # Start and terminate processes
    for w in workers:
        w.daemon = True
        w.start() 
    
    t4 = time.time() 
    print('Time to start workers: ', t4-t3, ' Num workers: ', num_workers)

    for i in range(3):
        task_q.put('play')

    for i in range(num_workers):
        task_q.put(None)

    agent.run()
    for w in workers:
        w.join()

    task_q.join()
    agent_in_q.put(None)
    agent_in_q.join()
    trainer_in_q.join()
    batch_gen_in_q.put(None)
    batch_gen_in_q.join()
    distributor_in_q.put(None)
    distributor_in_q.join()
    result_q.join()
    [pipe.join() for pipe in pipes]
    ##################################################
    
    print('\nTotal time elapsed: ', time.time()-t2)
