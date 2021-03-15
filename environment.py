class make_environment():
    def __init__(self, data) -> None:
        self.data = data 
        self.len = len(self.data)  
        self.idx = 0 

    def action_spec(self):
        return [0, 1, 2, 3]

    def observation_spec(self):
        return self.data.dtypes
    
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
        state = self.data.iloc[0].values
        return state, 0, False

