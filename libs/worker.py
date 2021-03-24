import sys
import time
import numpy as np
from multiprocessing import Process

from tensorflow.python.keras.layers.merge import concatenate

from libs.environment import Environment
from libs.datagen import DataGenerator
from libs.buffer import ReplayBuffer

class Worker(Process):
    
    def __init__(self, name, gen, task_q, agent_in_q, batch_gen_in_q, pipe, replay_buffer, result_q, v, settings):
        Process.__init__(self)
        self.name = name
        self.window_size = settings['window_size']
        self.normalization = settings['normalization']
        self.verbose = settings['verbose']
        self.env = Environment(DataGen=gen, settings=settings)
        self.v = v

        # Queues and Pipe
        self.task_q = task_q
        self.result_q = result_q
        self.agent_in_q = agent_in_q
        self.batch_gen_in_q = batch_gen_in_q
        self.replay_buffer = replay_buffer
        self.pipe = pipe

    def run(self):
        while True:
            next_task = self.task_q.get()
            if next_task is None:
                self.task_q.task_done()
                break
            
            if next_task == 'play':
                state = self.env.reset()
                done = False
                states = [[state[0]], [state[1]]]
                rewards = []
                advantages = []
                values = []
                
                while not done:
                    self.batch_gen_in_q.put(('actions', self.name, state, 0))
                    action, _, _ = self.pipe.get()
                    next_state, reward, done = self.env.step(action=action, epsilon=0)
                    self.batch_gen_in_q.put(('values', self.name, state, (action, reward, next_state, [done])))
                    _, advantage, value = self.pipe.get()
                    self.pipe.task_done()
                    
                    advantages.append(advantage)
                    values.append([value])
                    rewards.append(reward)
                    state = next_state
                    if not done:
                        states = (np.concatenate([states[0], [state[0]]], axis=0),
                                np.concatenate([states[1], [state[1]]], axis=0))

                advantages = np.reshape(advantages, np.shape(advantages))
                values = np.reshape(values, np.shape(values))
                self.replay_buffer.add(states, advantages, values, np.mean(rewards))
                if self.verbose == 2:
                    print('Episode reward: ', round(np.mean(rewards),2))

            elif next_task == 'train':
                samples = self.replay_buffer.get_samples(skewed=False)
                states, advantages, values = samples[0]
                self.agent_in_q.put(('train', self.name, states, (advantages, values)))

            self.task_q.task_done()
        print('Worker:', self.name,' is done!')
        self.v.value -=1

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