# 1D ResNet A2C

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.initializers import *
from keras.regularizers import *
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp

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
        x = Conv1D(16, kernel_size=7, strides=2, padding='same')(state)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        # ResNet Block
        b11 = Conv1D(32, kernel_size=1, strides=2, padding='same')(x)
        b11 = BatchNormalization()(b11)
        b11 = Activation('relu')(b11)
        b11 = Conv1D(32, kernel_size=3, strides=1, padding='same')(b11)
        b11 = BatchNormalization()(b11)
        b11 = Activation('relu')(b11)
        b11 = Conv1D(32, kernel_size=1, strides=1, padding='same')(b11)
        b11 = BatchNormalization()(b11)
        b11 = Add()([b11, Conv1D(32, kernel_size=1, strides=2)(x)])

        b11 = Activation('relu')(b11)

        # ResNet Block
        b12 = Conv1D(64, kernel_size=3, strides=2, padding='same')(b11)
        b12 = BatchNormalization()(b12)
        b12 = Activation('relu')(b12)
        b12 = Conv1D(64, kernel_size=3, strides=1, padding='same')(b12)
        b12 = BatchNormalization()(b12)
        b12 = Activation('relu')(b12)
        b12 = Conv1D(64, kernel_size=1, strides=1, padding='same')(b12)
        b12 = BatchNormalization()(b12)
        b12 = Activation('relu')(b12)
        b12 = Add()([b12, Conv1D(64, kernel_size=1, strides=2)(b11)])

        b12 = Activation('relu')(b12)

        # Output Block
        x = GlobalAvgPool2D()(b12)
        x = Dense(64)(x)
        x = Flatten()(x)

         # Input Block
        w = Conv1D(64, kernel_size=7, strides=2, padding='same')(game)
        w = BatchNormalization()(w)
        w = Activation('relu')(w)
        w = MaxPool2D(pool_size=3, strides=2, padding='same')(w)

        # ResNet Block
        b21 = Conv1D(32, kernel_size=3, strides=2, padding='same')(w)
        b21 = BatchNormalization()(b21)
        b21 = Activation('relu')(b21)
        b21 = Conv1D(32, kernel_size=3, strides=1, padding='same')(b21)
        b21 = BatchNormalization()(b21)
        b21 = Activation('relu')(b21)
        b21 = Conv1D(32, kernel_size=1, strides=1, padding='same')(b21)
        b21 = BatchNormalization()(b21)
        b21 = Activation('relu')(b21)
        b21 = Add()([b21, Conv1D(32, kernel_size=1, strides=2)(w)])

        b21 = Activation('relu')(b21)

        # ResNet Block
        b22 = Conv1D(64, kernel_size=3, strides=2, padding='same')(b21)
        b22 = BatchNormalization()(b22)
        b22 = Activation('relu')(b22)
        b22 = Conv1D(64, kernel_size=3, strides=1, padding='same')(b22)
        b22 = BatchNormalization()(b22)
        b22 = Activation('relu')(b22)
        b22 = Conv1D(64, kernel_size=1, strides=1, padding='same')(b22)
        b22 = BatchNormalization()(b22)
        b22 = Activation('relu')(b22)
        b22 = Add()([b22, Conv1D(64, kernel_size=1, strides=2)(b21)])

        b22 = Activation('relu')(b22)

        # Output Block
        w = GlobalAvgPool2D()(b22)
        w = Dense(64)(w)
        w = Flatten()(w)

        out = Concatenate()([w,x])

        actor_in = Dense(64, activation='relu')(out)
        actor_out = Dense(self.num_actions,
                         activation='softmax',
                         name='Dense_Actor_1')(actor_in)
        self.actor = Model(inputs=[state, game], outputs=actor_out)
        self.actor.compile(loss="categorical_crossentropy", optimizer=RMSProp(lr=self.lr_actor))


        critic_x = Dense(64, activation='relu', name='Dense_cr')(out)
        critic_out = Dense(self.num_values, activation='linear')(critic_x)
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