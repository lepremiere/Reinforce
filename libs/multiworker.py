import multiprocessing as mp
import numpy as np

class Worker(mp.Process):
    
    def __init__(self, name, task_queue, result_queue, agent_in_q, pipe):
        mp.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.agent_in_q = agent_in_q
        self.pipe = pipe
        self.name = name

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                print(self.pipe.qsize())
                self.task_queue.task_done()
                break
            ##################################################################################
            # DO SOMETHING

            ##################################################################################
            self.task_queue.task_done()
            print('...')
            self.result_queue.put(answer)
        print('Worker', self.name,'Queue = Done!')
        return


class Distributor(mp.Process):

    def __init__(self, distributor_in_q, pipes):
        mp.Process.__init__(self)
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

class BatchGenerator(mp.Process):

    def __init__(self, agent_in_q, agent_out_q, l):
        mp.Process.__init__(self)
        self.agent_in_q = agent_in_q
        self.agent_out_q = agent_out_q 
        self.l = l
    
    def run(self):
        i = 0
        ns, samples = [], []
        while True:
            if i+750 > self.l:
                n, next_sample = self.agent_in_q.get()
                self.agent_out_q.put([np.array([n]), np.array([next_sample])])
                self.agent_in_q.task_done()
                print('done Batches')
            else:
                n, next_sample = self.agent_in_q.get()
                self.agent_in_q.task_done()
                n = int(n)
                ns.append(n)
                samples.append(next_sample)
                if len(ns) == 22:
                    self.agent_out_q.put([np.array(ns), np.array(samples)])
                    ns, samples = [], []
            i += 1
        print('Prediciton Queue = Done!')
        return