import sys
import time
import numpy as np
from multiprocessing import Process

from libs.environment import Environment
from libs.datagen import DataGenerator
from libs.buffer import ReplayBuffer

class Worker(Process):
    
    def __init__(self, name, gen, buffer, task_q, result_q, agent_in_q, batch_gen_in, distributor_pipe, settings):
        Process.__init__(self)
        self.name = name
        self.window_size = settings['window_size']
        self.env = Environment(DataGen=gen, normalization=settings['normalization'], verbose=settings['verbose'])

        # Queues and Pipe
        self.task_q = task_q
        self.result_q = result_q
        self.agent_in_q = agent_in_q
        self.batch_gen_in = batch_gen_in
        self.distributor_pipe = distributor_pipe

    def run(self):
        while True:
            next_task = self.task_q.get()
            if next_task is None:
                self.task_q.task_done()
                break

            output = self.name # self.play_episode()
            self.batch_gen_in.put(output)

            self.task_queue.task_done()
        print('Worker:', self.name,' is done!')

    def play_episode(self, day):

        total_rewards = []
        trades = []
        profit = []
        total_reward = 0.0
        state = self.env.reset()
        dim = np.shape(self.env.day)[1]-1   
        state[0] = np.reshape(np.array(state[0]), (1, self.window_size, dim))

        while True:
            batch_gen_in.put(state)
            action = distributor_pipe.get()
            next_state, reward, done = env.step(action, agent.epsilon)
            next_state[0] = np.reshape(next_state[0], (1, window_size, dim))

            advantage, value = agent.get_values(state, action, reward, next_state, done)
            self.buffer.add(state, advantage, value, [reward])

            total_reward += reward
            state = next_state

            if done:
                profit.append(np.sum(env.trades['Profit']))
                trades.append(env.trades)
                total_rewards.append(total_reward)
                mean_total_rewards = np.mean(total_rewards[-min(len(total_rewards), 10):])

                print("Episode: ", episode+1,
                    " Total Reward: ", total_reward,
                    " Mean: ", mean_total_rewards,
                    " Time: ", round(time.time()-t,2), " s")
                break
        return 1

if __name__ == "__main__":
    pass