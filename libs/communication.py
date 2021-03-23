import numpy as np
from multiprocessing import Process

class BatchGenerator(Process):

    def __init__(self, in_q, out_q, k, settings):
        Process.__init__(self)
        self.batch_gen_in_q = in_q
        self.agent_in_q = out_q 
        self.k = k
    
    def run(self):
        states1 = []
        states2 = []

        ns = []
        while True:
            n, state = self.batch_gen_in_q.get()
            if state == None:
                break
            states1.append(state[0])
            states2.append(state[1])
            ns.append(n)
            if len(ns) >= self.k:
                self.agent_in_q.put((ns, states1, states2))
                states1 = []
                states2 = []
                ns = []
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
            ns, policies = self.distributor_in_q.get()
            print(ns,policies)
            if policies == None:
                break
            for i in range(len(ns)):
                actions = np.random.choice(policies[i])
                self.pipes[int(ns[i])].put(actions[i])
            self.distributor_in_q.task_done()
        print('Prediciton Queue is done!')
        return