import numpy as np

class environment():
    def __init__(self, days) -> None:
        self.days = days
        self.len = len(self.days)  
        self.idx = 0 
        self.action_space_n = 4
        self.observation_space_n = self.days[0].shape[1]

    def action_space(self):
        return [0, 1, 2, 3]

    def observation_space(self):
        return self.days[0].dtypes
    
    def step(self, action):

        if self.idx < self.len-1:
            self.idx += 1
            done = False
        else:
            done = True

        state = self.data.iloc[self.idx].values
        reward = 1
        
        return state, reward, done

    def reset(self):
        self.idx = 0
        i = np.random.choice(range(len(self.days)))
        self.data = self.days[i]
        state = self.data.iloc[0].values
        return state

