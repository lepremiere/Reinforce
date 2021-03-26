import numpy as np
import time
from multiprocessing import Process

class BatchGenerator(Process):

    def __init__(self, in_q, out_q, val, settings):
        Process.__init__(self)
        self.batch_gen_in_q = in_q
        self.agent_in_q = out_q 
        self.val = val
        self.num_workers = settings['num_workers']
        self.verbose = settings['verbose']
    
    def run(self):
        actions, rewards = [], []
        states1_val, states2_val = [], []
        next_states1, next_states2 = [], []
        ns_vals, dones = [], []
        states1, states2 = [], []
        ns_actions = []

        while True:
            
            if np.sum(self.val[:]) == 0:
                    break
            try:
                type, n, state, val_stuff = self.batch_gen_in_q.get(timeout=0.0) 
                self.batch_gen_in_q.task_done()

                if type == 'actions':
                    states1.append(state[0])
                    states2.append(state[1])
                    ns_actions.append(n) 

                elif type ==  'values':
                    action, reward, next_state, done = val_stuff
                    actions.append(action)
                    rewards.append(reward)
                    states1_val.append(state[0])
                    states2_val.append(state[1])
                    next_states1.append(next_state[0])
                    next_states2.append(next_state[1])
                    dones.append(done)
                    ns_vals.append(n)   

                elif type == 'train':
                    self.agent_in_q.put((type, None, state, val_stuff))                      

            except: 
                playing_worker = np.sum([v == 1 for v in self.val])
                number = playing_worker/3

                if len(ns_actions) >= number:
                    states1 = np.reshape(states1, np.shape(states1))
                    states2 = np.reshape(states2, np.shape(states2))
                    self.agent_in_q.put(('actions', ns_actions, (states1, states2), None))
                    states1, states2, ns_actions = [], [], []

                    if self.verbose == 3:
                        print('Actions generator flushed!')   
                            
                elif len(ns_vals) >= number:
                    states1_val = np.reshape(states1_val, np.shape(states1_val))
                    states2_val = np.reshape(states2_val, np.shape(states2_val))
                    next_states1 = np.reshape(next_states1, np.shape(next_states1))
                    next_states2 = np.reshape(next_states2, np.shape(next_states2))
                    actions = np.reshape(actions, np.shape(actions))
                    rewards = np.reshape(rewards, np.shape(rewards))
                    dones = np.reshape(dones, np.shape(dones))
                    self.agent_in_q.put(('values', ns_vals, (states1_val, states2_val), (actions, rewards, (next_states1, next_states2), dones)))
                    actions, rewards = [], []
                    states1_val, states2_val = [], []
                    next_states1, next_states2 = [], []
                    ns_vals, dones = [], []

                    if self.verbose == 3:
                        print('Vals generator flushed!')                
                
        print('Batch Generator is done!')
    
class Distributor(Process):

    def __init__(self, in_q, pipes, val, settings):
        Process.__init__(self)
        self.distributor_in_q = in_q
        self.pipes = pipes 
        self.val = val
    
    def run(self):
        while True:
            try:
                ns, actions, advantages, values = self.distributor_in_q.get(timeout=0.1)
                self.distributor_in_q.task_done()
                if len(ns) == 1:
                    self.pipes[int(ns[0])].put((actions[0], advantages[0], values[0]))
                else: 
                    for i in range(len(ns)):
                        self.pipes[int(ns[i])].put((actions[i], advantages[i], values[i]))
            except:
                if np.sum(self.val[:]) == 0:
                    break
            
        print('Distributor is done!')
