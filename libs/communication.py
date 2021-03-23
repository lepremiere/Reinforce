import numpy as np
from multiprocessing import Process

class BatchGenerator(Process):

    def __init__(self, in_q, out_q, k, settings):
        Process.__init__(self)
        self.batch_gen_in_q = in_q
        self.agent_in_q = out_q 
        self.k = k
        self.verbose = settings['verbose']
    
    def run(self):
        states1 = []
        states2 = []
        ns = []
        while True:
            n, state = self.batch_gen_in_q.get()
            if n == None:
                states1 = np.reshape(states1, np.shape(states1))
                states2 = np.reshape(states2, np.shape(states2))
                self.agent_in_q.put((ns, states1, states2))
                self.batch_gen_in_q.task_done() 
                break  
            states1.append(state[0])
            states2.append(state[1])
            ns.append(n)
            if len(ns) >= self.k:
                states1 = np.reshape(states1, np.shape(states1))
                states2 = np.reshape(states2, np.shape(states2))
                self.agent_in_q.put((ns, states1, states2))
                states1 = []
                states2 = []
                ns = []
                if self.verbose > 1:
                    print('Batch generated!')
            self.batch_gen_in_q.task_done()     
        print('\nBatch Generator is done!')
        return

class Distributor(Process):

    def __init__(self, in_q, pipes, settings):
        Process.__init__(self)
        self.distributor_in_q = in_q
        self.pipes = pipes 
    
    def run(self):
        while True:
            ns, actions = self.distributor_in_q.get()
            self.distributor_in_q.task_done()
            if ns == None:
                break
            for i in range(len(ns)):
                self.pipes[int(ns[i])].put(actions[i])
        print('Prediciton Queue is done!')
        return