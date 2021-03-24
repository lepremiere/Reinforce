import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Value

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

    settings = {'num_workers': 20,
                'batch_size': 64,
                'num_episodes': 1,
                'window_size': 100,
                'normalization': False,
                'buffer_size': int(1e4),
                'verbose': 1}

    num_workers = settings['num_workers']
    v = Value('i', num_workers)

    # Queues
    task_q = mp.JoinableQueue()
    agent_in_q = mp.JoinableQueue()
    batch_gen_in_q = mp.JoinableQueue()
    distributor_in_q = mp.JoinableQueue()
    pipes = [mp.JoinableQueue() for _ in range(num_workers)]

    # Processes
    gen = DataGenerator(symbol="SP500_M1", fraction=[1, 1e4], settings=settings)
    batch_gen = BatchGenerator(in_q=batch_gen_in_q, out_q=agent_in_q, k=v, settings=settings)
    distributor = Distributor(in_q=distributor_in_q, pipes=pipes, val=v, settings=settings)
    buffer = ReplayBuffer(settings)
    env = Environment(DataGen=gen, settings=settings)
    agent = Agent(env=env, in_q=agent_in_q, out_q=distributor_in_q, val=v, settings=settings)
    workers = [Worker(name=str(i),
                      gen=gen,
                      task_q=task_q,
                      agent_in_q=agent_in_q,
                      batch_gen_in_q=batch_gen_in_q,
                      pipe=pipes[i],
                      replay_buffer=buffer,
                      result_q=1,
                      v=v,
                      settings=settings) \
                for i in range(num_workers)]
    #################################################
    # Start and terminate processes
    agent.daemon = True
    distributor.daemon = True
    batch_gen.daemon = True
    agent.start()
    distributor.start()
    batch_gen.start() 

    for w in workers:
        w.daemon = True
        w.start() 
    
    t4 = time.time() 
    print('Time to start sampling: ', round(t4-t2, 2), ' s,  Num workers: ', num_workers)

    for i in range(10):
        for j in range(50):
            task_q.put('play')
        for i in range(5):
            task_q.put('train')    

    ##################################################
    # Shut down
    for i in range(num_workers):
        task_q.put(None)

    # agent.run()
    for w in workers:
        w.join()
    batch_gen.join()
    distributor.join()

    agent_in_q.put((None, None, None, None))
    agent_in_q.close()
    task_q.close()
    batch_gen_in_q.close()
    distributor_in_q.close()

    for pipe in pipes:
        pipe.close()
    ##################################################
    
    print('\nTotal time elapsed: ', round(time.time()-t2, 2))
