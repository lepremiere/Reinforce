import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Process

from libs.environment import Environment

class Worker(Process):
    
    def __init__(self, name,
                 data_gen_in_q, data_gen_out_q,
                 buffer_in_q, buffer_out_q,
                 task_q,
                 agent_in_q,
                 batch_gen_in_q,
                 pipe,
                 replay_buffer,
                 news_q,
                 market,
                 val,
                 settings):
        Process.__init__(self)
        self.name = name
         # Queues and Pipe
        self.data_gen_in_q = data_gen_in_q
        self.data_gen_out_q = data_gen_out_q
        self.buffer_in_q = buffer_in_q
        self.buffer_out_q = buffer_out_q
        self.task_q = task_q
        self.news_q = news_q
        self.agent_in_q = agent_in_q
        self.batch_gen_in_q = batch_gen_in_q
        self.replay_buffer = replay_buffer
        self.pipe = pipe
        self.val = val

        self.env = Environment(in_q=self.data_gen_in_q,
                               out_q=self.data_gen_out_q,
                               news_q=news_q,
                               market=market,
                               settings=settings)

        self.window_size = settings['window_size']
        self.normalization = settings['normalization']
        self.gamma = settings['gamma']
        self.verbose = settings['verbose']

    def run(self):
        while True:
            next_task = self.task_q.get()
            self.task_q.task_done()

            if next_task is None:
                break
            
            if next_task == 'play':
                state = self.env.reset()
                done = False
                actions = []
                states1 = []
                states2 = []
                rewards = []
                advantages = []
                values = []
                dones = []
                
                while not done:
                    self.batch_gen_in_q.put(('play', self.name, state, 0))
                    action, value = self.pipe.get()
                    self.pipe.task_done()
                    next_state, reward, done = self.env.step(action=action)

                    actions.append(action)
                    rewards.append(reward)
                    values.append(value)
                    dones.append(done)
                    states1.append(state[0])
                    states2.append(state[1])
                    state = next_state

                # Discounted rewards
                discounted_rewards = self.discount_reward(actions, rewards, dones, values, self.env.tracker)
                advantages = np.zeros(shape=(len(dones), 3))

                for i in range(len(dones)):
                    if dones[i]:
                        advantages[i][actions[i]] = discounted_rewards[i] - values[i]
                        values[i][0] = discounted_rewards[i]
                    else:
                        advantages[i][actions[i]] = (discounted_rewards[i] + self.gamma * values[i+1]) - values[i]
                        values[i][0] = discounted_rewards[i] + self.gamma * values[i+1]  

                discounted_rewards = np.reshape(discounted_rewards, np.shape(discounted_rewards)) 
                states1 = np.reshape(states1, np.shape(states1))
                states2 = np.reshape(states2, np.shape(states2))
                values = np.reshape(values, np.shape(values))
                advantages = np.reshape(advantages, np.shape(advantages))
                self.buffer_in_q.put(('add_sample', ((states1, states2), advantages, values, np.mean(discounted_rewards))))
                
                if self.verbose == 1:
                    # Overwatch 
                    try:
                        duration = self.env.trades['Duration']
                    except:
                        duration = 0
                    message = ('Episode_end', [len(self.env.day),
                                               round(np.sum(discounted_rewards),2),
                                               round(np.min(discounted_rewards),2),
                                               round(np.max(discounted_rewards),2),
                                               round(np.sum(self.env.trades['Profit']), 2),
                                               round(np.mean(duration), 2),
                                               round(len(self.env.trades))])
                    self.news_q.put(message)
                del states1
                del states2
                del values
                del advantages
                del dones
                del rewards
                del discounted_rewards
            elif next_task == 'train':
         
                if self.verbose == 1:
                    print('Training...')
                self.buffer_in_q.put(('get_samples', 0))
                samples = self.buffer_out_q.get()
                self.buffer_out_q.task_done()
                states, advantages, values = samples[0]
                self.batch_gen_in_q.put(('train', self.name, states, (advantages, values)))
             

        print('Worker:', self.name,' is done!')
        self.val[int(self.name)] = 0
    
    def get_statistics(self):
        pass

    def discount_reward(self, actions, rewards, dones, values, tracker):
        # Close Profit on Open position state
        # for i in range(len(self.env.trades)):
        #     idx, idx2 = self.env.trades['Openindex'].iloc[i], self.env.trades['Closeindex'].iloc[i]
            # # if self.env.trades['Duration'].iloc[i] < 6 and self.env.trades['Profit'].iloc[i] < 0:
            # #     rewards[idx] += tracker['Current_Profit'].iloc[idx2] - 100
            # # elif self.env.trades['Duration'].iloc[i] < 10 and self.env.trades['Profit'].iloc[i] > 50:
            # #     rewards[idx] += tracker['Current_Profit'].iloc[idx2] + 100
            # rewards[idx][actions[i]] += tracker['Current_Profit'].iloc[idx2]
        # Debug
        # a = pd.DataFrame(data={'rewards': rewards, 'actions': actions})
        # b = tracker
        # b.to_csv('./2_storage/tracker.csv')        
        # a.to_csv('./2_storage/action_rewards.csv')       

        # discounted_rewards = [np.sum(0.9**np.arange(len(rewards[i:]))*rewards[i:]) for i in range(len(rewards))]
        discounted_rewards = rewards
        
        return discounted_rewards

if __name__ == "__main__":
    pass