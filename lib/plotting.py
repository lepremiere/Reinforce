import keras 
import matplotlib.pyplot as plt
from IPython.display import clear_output

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

        self.n = 50

    def on_epoch_end(self, epoch, logs={}):
        if epoch > 2:
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.i += 1

            clear_output(wait=True)
            plt.plot(self.x[-self.n:], self.losses[-self.n:], label="loss")
            plt.plot(self.x[-self.n:], self.val_losses[-self.n:], label="val_loss")
            plt.legend()
            plt.show()

from numpy import fft
    


def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0-last_days
    
    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x = range(0, len(dataset))
    
    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(x,dataset['ma7'],label='MA 7', color='g',linestyle='--')
    plt.plot(x,dataset['Close'],label='Closing Price', color='b')
    plt.plot(x,dataset['ma21'],label='MA 21', color='r',linestyle='--')
    plt.plot(x,dataset['upper_band'],label='Upper Band', color='c')
    plt.plot(x,dataset['lower_band'],label='Lower Band', color='c')
    plt.fill_between(x, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for Goldman Sachs - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(x,dataset['MACD'],label='MACD', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(x,dataset['momentum'],label='Momentum', color='b',linestyle='-')

    plt.legend()
    plt.show()