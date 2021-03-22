import sys
import time
import numpy as np
from multiprocessing import Process

from libs.environment import Environment
from libs.datagen import DataGenerator
from libs.buffer import ReplayBuffer

class Worker(Process):
    
    def __init__(self, name, gen, task_queue, settings):
        Process.__init__(self)
        self.env = Environment(DataGen=gen, normalization=settings['normalization'], verbose=settings['verbose'])
        self.task_queue = task_queue
        self.name = name


    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                self.task_queue.task_done()
                break
            state = self.env.reset()
            done = True
            while not done:
                next_state, _, done = self.env.step(action=1, epsilon=0)
            ##################################################################################
            # DO SOMETHING
            # tu hier alles rein was in der test function ist baljat!
            ##################################################################################
            self.task_queue.task_done()
        print('Worker:', self.name,' is done!')

if __name__ == "__main__":
    pass