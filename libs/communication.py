import numpy as np
import time
from multiprocessing import Process

class BatchGenerator(Process):

    def __init__(self, in_q, out_q, news_q, val, settings):
        Process.__init__(self)
        self.batch_gen_in_q = in_q
        self.agent_in_q = out_q 
        self.news_q = news_q
        self.val = val
        self.num_workers = settings['num_workers']
        self.verbose = settings['verbose']
    
    def run(self):
        states1, states2 = [], []
        ns_actions = []

        while True:
            
            if np.sum(self.val[:]) == 0:
                    break
            try:
                type, n, state, val_stuff = self.batch_gen_in_q.get(timeout=0.00) 
                self.batch_gen_in_q.task_done()

                if type == 'play':
                    states1.append(state[0])
                    states2.append(state[1])
                    ns_actions.append(n)

                elif type == 'train':
                    self.agent_in_q.put((type, None, state, val_stuff))    

                del(type)
                del(n)
                del(state)
                del(val_stuff)                  

            except: 
                playing_worker = np.sum([v == 1 for v in self.val])
                number = playing_worker

                if len(ns_actions) >= number:
                    states1 = np.reshape(states1, np.shape(states1))
                    states2 = np.reshape(states2, np.shape(states2))
                    # print(np.shape(states1)[0])
                    self.agent_in_q.put(('play', ns_actions, (states1, states2), 0))
                    del(states1)
                    del(states2)
                    del(ns_actions)
                    states1, states2, ns_actions = [], [], []

                    if self.verbose == 3:
                        print('Com flushed!')                 
                
        print('Batch Generator is done!')
    
class Distributor(Process):

    def __init__(self, in_q, pipes, news_q, val, settings):
        Process.__init__(self)
        self.distributor_in_q = in_q
        self.pipes = pipes 
        self.news_q = news_q
        self.val = val
    
    def run(self):
        while True:
            try:
                ns, actions, values = self.distributor_in_q.get(timeout=0.1)
                self.distributor_in_q.task_done()
                if len(ns) == 1:
                    self.pipes[int(ns[0])].put((actions[0], values[0]))
                else: 
                    for i in range(len(ns)):
                        self.pipes[int(ns[i])].put((actions[i], values[i]))
                del ns 
                del actions
                del values
            except:
                if np.sum(self.val[:]) == 0:
                    break
            
        print('Distributor is done!')
