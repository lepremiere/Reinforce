import numpy as np
from multiprocessing import Process

class BatchGenerator(Process):

    def __init__(self, batch_gen_in_q, agent_in_q, l):
        Process.__init__(self)
        self.batch_gen_in_q = batch_gen_in_q
        self.agent_in_q = agent_in_q 
        self.l = l
    
    def run(self):
        i = 0
        ns, samples = [], []
        while True:
            break
        print('Prediciton Queue = Done!')
        return

class Distributor(Process):

    def __init__(self, distributor_in_q, pipes):
        Process.__init__(self)
        self.distributor_in_q = distributor_in_q
        self.pipes = pipes 
    
    def run(self):
        while True:
            ns, next_samples = self.distributor_in_q.get()
            for i in range(len(ns)):
                self.pipes[ns[i]].put(next_samples[i])
            self.distributor_in_q.task_done()
        print('Prediciton Queue = Done!')
        return