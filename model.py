from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.initializers import *
import numpy as np
from tensorflow import concat

class NN(Model):
    def __init__(self, num_observations, num_actions, num_values, lr_actor, lr_critic):
        super(NN, self).__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_values = num_values
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        state = Input(shape=self.num_observations, name='Input')

        x = Conv1D(64, 16, name='Dense_1', activation='relu')(state)
        x = Dense(64, name='Dense_2', activation='relu')(x)
        x = Dense(8, name='Dense_3', activation='relu')(x)
        x = Flatten()(x)

        actor_out = Dense(self.num_actions,
                         activation='softmax',
                         name='Dense_Actor_1')(x)
        self.actor = Model(inputs=state, outputs=actor_out)
        self.actor.summary()
        self.actor.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.lr_actor))

        critic_x = Dense(64, name='Dense_cr')(x)
        critic_x = Dense(self.num_values)(critic_x)
        critic_out = Activation("linear")(critic_x)
        self.critic = Model(inputs=state, outputs=critic_out)
        self.critic.summary()
        self.critic.compile(loss="mse", optimizer=Adam(lr=self.lr_critic))        
    
    # Actor Functions
    def train_actor(self, states, game, advantages):
        self.actor.fit(states, advantages, verbose=0, epochs=1)

    def predict_actor(self, game, states):
        return self.actor(states)

    # Critic Functions
    def train_critic(self, states, game, values):
        self.critic.fit(states, values, verbose=0, epochs=1)

    def predict_critic(self, states, game):
        return self.critic(states)