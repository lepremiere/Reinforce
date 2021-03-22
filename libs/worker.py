import multiprocessing as mp
import numpy as np
from numpy.core.fromnumeric import shape

from libs.environment import Environment
from libs.datagen import DataGenerator
from libs.buffer import ReplayBuffer

class Worker(mp.Process):
    
    def __init__(self, name, gen, task_queue, settings):
        mp.Process.__init__(self)
        self.env = Environment(DataGen=gen, normalization=settings['normalization'], verbose=settings['verbose'])
        self.task_queue = task_queue
        self.name = name

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                self.task_queue.task_done()
                break
            
            _ = self.env.reset()
            done = True
            while not done:
                _, _, done = self.env.step(action=0, epsilon=0)
            self.task_queue.task_done()
            ##################################################################################
            # DO SOMETHING
            # tu hier alles rein was in der test function ist baljat!
            ##################################################################################
        print('Worker:', self.name,' is done!')
        return

if __name__ == "__main__":
    pass