# Autoencoder A2C

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.initializers import *
from keras.regularizers import *
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp
from keras.layers import Bidirectional
from tensorflow.python.keras.layers import CuDNNLSTM

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass

class NN(Model):
    def __init__(self, num_observations, num_actions, num_values, game_size, lr_actor, lr_critic, settings):
        super(NN, self).__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_values = num_values
        self.game_size = game_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.verbose = settings['verbose']
        self.epochs = settings['training_epochs']

        state = Input(shape=self.num_observations, name='Input')
        game = Input(shape=self.game_size, name='Input_game')

        # Input Block
        x = BatchNormalization()(state)
        l1 = CuDNNLSTM(64, return_sequences=True)(x)
        x = Dense(128, activation='relu')(l1)
        x = Dense(64, activation='relu')(x)
        x = CuDNNLSTM(64, return_sequences=True)(x)
        x = Add()([x, l1])
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        l1 = CuDNNLSTM(64, return_sequences=True)(x)
        x = Dense(128, activation='relu')(l1)
        x = Dense(64, activation='relu')(x)
        x = CuDNNLSTM(64, return_sequences=True)(x)
        x = Add()([x, l1])
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        l1 = CuDNNLSTM(64, return_sequences=True)(x)
        x = Dense(128, activation='relu')(l1)
        x = Dense(64, activation='relu')(x)
        x = CuDNNLSTM(64, return_sequences=False)(x)
        x = Activation('relu')(x)
        x = Dense(128, activation='relu')(x) 
        x = Dense(64, activation='relu')(x) 
        x = Dense(32, activation='relu')(x)  
        x = Dense(16, activation='relu')(x)  
        x = Flatten()(x)
        out = Concatenate()([x, game])

        actor_in = Dense(64, activation='relu')(out)
        actor_in = Dense(32, activation='relu')(actor_in)
        actor_in = Dense(16, activation='relu')(actor_in)
        actor_out = Dense(self.num_actions,
                         activation='softmax',
                         name='Dense_Actor_1')(actor_in)
        self.actor = Model(inputs=[state, game], outputs=actor_out)
        self.actor.compile(loss="categorical_crossentropy", optimizer=RMSProp(lr=self.lr_actor))

        critic_in = Dense(64, activation='relu')(out)
        critic_in = Dense(32, activation='relu')(critic_in)
        critic_in = Dense(16, activation='relu')(critic_in)
        critic_out = Dense(self.num_values, activation='linear')(critic_in)
        self.critic = Model(inputs=[state, game], outputs=critic_out)
        self.critic.compile(loss="mse", optimizer=RMSprop(lr=self.lr_critic)) 

        # plot_model(self.actor, to_file='./1_model/Actor.png',
        #     show_shapes=True, show_dtype=True,
        #     show_layer_names=True)
        # plot_model(self.critic, to_file='./1_model/Critic.png',
        #     show_shapes=True, show_dtype=True,
        #     show_layer_names=True) 
        if self.verbose > 0:
            self.actor.summary()
            self.critic.summary()

if __name__ == '__main__':
    pass