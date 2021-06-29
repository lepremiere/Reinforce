import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass

df = pd.read_csv('D:/Data/SP500_M5_live_TA.csv')

if df.columns[0] == 'Date':
    df.columns = map(str.lower, df.columns)
time = pd.to_datetime(df.date)
df.date = time

logits = df.columns[df.isnull().sum() > len(df)/3]
for logit in logits:
    print('Popped: ', logit, ' with ', df.loc[:, logit].isnull().sum(), ' NaNs')
    df.pop(logit)
print(df.columns[ df.isnull().sum() != 0])
n = df.isnull().sum().max()
df = df.loc[df.isnull().any(axis=1)==False, :]

df.insert(loc=1, column='minute', value=time.dt.minute.astype(np.float16))
df.insert(loc=1, column='hour', value=time.dt.hour.astype(np.float16))
df.insert(loc=1, column='day', value=time.dt.dayofweek.astype(np.float16))
df.insert(loc=1, column='month', value=time.dt.month.astype(np.float16))
print(df)
fun = []
fun1 = []
close = df.close.to_numpy()
w = 12
for i in range(len(close)-1):
    if i == len(close) - w:
        w -= 1
    max_up = np.argmax(close[i:i+w])
    max_down = np.argmin(close[i:i+w])
    ratio_up = (close[i+max_up] - close[i])/(close[i] - close[i+max_down] + 1)
    fun.append(close[i+max_up] - close[i])
    max_down = np.argmin(close[i:i+w])
    max_up = np.argmax(close[i:i+w])
    ratio_down = (close[i] - close[i+max_down])/(close[i+max_up] - close[i] + 1)
    fun1.append(close[i] - close[i+max_down])
fun.append(0)
fun1.append(0)

window_size = 100
test_split = 0.2

X_train = df.iloc[0:-int(len(df)*test_split)]
X_test = df.iloc[-int(len(df)*test_split):]
fun_train = fun[0:-int(len(df)*test_split)]
fun_test = fun[-int(len(df)*test_split):]
fun1_train = fun1[0:-int(len(df)*test_split)]
fun1_test = fun1[-int(len(df)*test_split):]

X_tr = X_train.iloc[:,1:].to_numpy()
X_te = X_test.iloc[:,1:].to_numpy()
print(np.shape(X_tr))

X_train_chopped = np.array([X_tr[i-window_size:i]  for i in range(window_size+1,len(X_tr))]).astype(np.float32)
y_train = np.column_stack((fun_train[window_size:],fun1_train[window_size:])).astype(np.float32)

X_test_chopped = np.array([X_te[i-window_size:i] for i in range(window_size+1,len(X_te))]).astype(np.float32)
y_test = np.column_stack((fun_test[window_size:],fun1_test[window_size:])).astype(np.float32)

from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import *
model = Sequential()
model.add(Input(shape=(window_size, df.shape[1]-1)))
model.add(BatchNormalization(trainable=False))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1024, kernel_regularizer=L2(l2=0.01), activation='relu'))
model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(128, kernel_regularizer=L2(l2=0.01), activation='relu'))
model.add(Dense(64, kernel_regularizer=L2(l2=0.01), activation='relu'))
model.add(Dense(32, kernel_regularizer=L2(l2=0.01), activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='linear'))
model.compile(optimizer=RMSprop(lr=0.001), loss=Huber(delta=1))
model.summary()

model.fit(x=X_train_chopped, y=y_train,
        validation_split=0.2,
        shuffle=True,
        epochs=20, batch_size=128, verbose=True)
y_pred_train = model.predict(x=X_train_chopped)
y_pred_test = model.predict(x=X_test_chopped)

thresh = 15
fig, ax = plt.subplots(nrows=3, figsize=(15, 15), sharex=True)
ax[0].plot(X_train.date, X_train.close, color='b')
ax[0].plot(X_test.date, X_test.close, color='g')
ax[0].axvline(X_train.date.iloc[window_size], color='k')

ax[1].plot(X_train.date, fun_train, color='b')
ax[1].plot(X_test.date, fun_test, color='g')
ax[1].axhline(thresh, color='k')

ax[1].plot(X_train.date[window_size:-1],y_pred_train[:,0], color='r')
ax[1].plot(X_test.date[window_size:-1],y_pred_test[:,0], color='m')

ax[2].plot(X_train.date, fun1_train, color='b')
ax[2].plot(X_test.date, fun1_test, color='g')
ax[2].axhline(thresh, color='k')

ax[2].plot(X_train.date[window_size:-1],y_pred_train[:,1], color='r')
ax[2].plot(X_test.date[window_size:-1],y_pred_test[:,1], color='m')

plt.setp(ax[2].get_xticklabels(), rotation=45)
plt.show()

model.save('model.h5')
