import numpy as np
from multiprocessing import Process

class BatchGenerator(Process):

    def __init__(self, in_q, out_q, k, settings):
        Process.__init__(self)
        self.batch_gen_in_q = in_q
        self.agent_in_q = out_q 
        self.k = k
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
            try:
                type, n, state, val_stuff = self.batch_gen_in_q.get(timeout=5) 
                self.batch_gen_in_q.task_done()
                if self.k.value == 0:
                    break              
                if type == 'actions':
                    states1.append(state[0])
                    states2.append(state[1])
                    ns_actions.append(n)

                    if len(ns_actions) >= 10:
                        states1 = np.reshape(states1, np.shape(states1))
                        states2 = np.reshape(states2, np.shape(states2))
                        states = (states1, states2)
                        self.agent_in_q.put((type, ns_actions, states, None))
                        states1, states2 = [], []
                        ns_actions = []

                        if self.verbose > 1:
                            print('Batch generated!')  

                if type ==  'values':
                    action, reward, next_state, done = val_stuff
                    actions.append(action)
                    rewards.append(reward)
                    states1_val.append(state[0])
                    states2_val.append(state[1])
                    next_states1.append(next_state[0])
                    next_states2.append(next_state[1])
                    dones.append(done)
                    ns_vals.append(n)

                    if len(ns_vals) >= 10:
                        states1_val = np.reshape(states1_val, np.shape(states1_val))
                        states2_val = np.reshape(states2_val, np.shape(states2_val))
                        next_states1 = np.reshape(next_states1, np.shape(next_states1))
                        next_states2 = np.reshape(next_states2, np.shape(next_states2))
                        actions = np.reshape(actions, np.shape(actions))
                        rewards = np.reshape(rewards, np.shape(rewards))
                        dones = np.reshape(dones, np.shape(dones))
                        self.agent_in_q.put((type, ns_vals, (states1_val, states2_val), (actions, rewards, (next_states1, next_states2), dones)))
                        actions, rewards = [], []
                        states1_val, states2_val = [], []
                        next_states1, next_states2 = [], []
                        ns_vals, dones = [], []

            except:
                if self.k.value == self.num_workers:
                    if self.k.value == 0:
                        break
                elif self.k.value != self.num_workers:
                    if self.k.value == 0:
                        break
                    # Flush actor batches
                    if len(states1) > 0:
                        states1 = np.reshape(states1, np.shape(states1))
                        states2 = np.reshape(states2, np.shape(states2))
                        self.agent_in_q.put((type, ns_actions, (states1, states2), None))
                        states1, states2, ns_actions = [], [], []

                    # Flush critic batches
                    if len(states1_val) > 0:
                        states1_val = np.reshape(states1_val, np.shape(states1_val))
                        states2_val = np.reshape(states2_val, np.shape(states2_val))
                        next_states1 = np.reshape(next_states1, np.shape(next_states1))
                        next_states2 = np.reshape(next_states2, np.shape(next_states2))
                        actions = np.reshape(actions, np.shape(actions))
                        rewards = np.reshape(rewards, np.shape(rewards))
                        dones = np.reshape(dones, np.shape(dones))
                        self.agent_in_q.put((type, ns_vals, (states1_val, states2_val), (actions, rewards, (next_states1, next_states2), dones)))
                        actions, rewards = [], []
                        states1_val, states2_val = [], []
                        next_states1, next_states2 = [], []
                        ns_vals, dones = [], []

                    self.num_workers -= 1
                    if self.verbose == 1:
                        print('Batch generator flushed')
                        print('Reducing batch size to ', self.num_workers)
                
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
                ns, actions, advantages, values = self.distributor_in_q.get(timeout=1)
                self.distributor_in_q.task_done()
                if len(ns) == 1:
                    self.pipes[int(ns[0])].put((actions[0], advantages[0], values[0]))
                else: 
                    for i in range(len(ns)):
                        self.pipes[int(ns[i])].put((actions[i], advantages[i], values[i]))
            except:
                if self.val.value == 0:
                    break
            
        print('Distributor is done!')
