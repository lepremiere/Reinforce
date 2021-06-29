from pykalman import KalmanFilter
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('D:/Data/SP500_H1_TA.csv',skiprows=range(1,20000))
plt.plot(df.close)

for i in [6,12]:
    df[f'MA_{i}'] = df.close.rolling(i).mean()
    plt.plot(df[f'MA_{i}'], 'g')
for i in [.01, .1]:
    kf = KalmanFilter(transition_matrices = [1],
                    observation_matrices = [1],
                    initial_state_mean = 0,
                    initial_state_covariance = 1,
                    observation_covariance=1,
                    transition_covariance=i)
    state_means, _ = kf.filter(df.close)
    plt.plot(state_means, '--r')
plt.show()