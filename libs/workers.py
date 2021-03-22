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
                self.task_queue.task_done()
                break
            ##################################################################################
            # DO SOMETHING
            # tu hier alles rein was in der test function ist baljat!

            ##################################################################################
            self.task_queue.task_done()
            print('...')
            self.result_queue.put(answer)
        print('Worker', self.name,'Queue = Done!')
        return