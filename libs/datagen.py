import numpy as np 
import pandas as pd 

class DataGenerator():
    def __init__(self, symbol: str, fraction=[0,0], window_size=100, test_split=0.2) -> None:
        self.df = pd.read_csv("D:/Data/" + symbol + ".csv",
                                header=0,
                                skiprows=range(1,int(fraction[0])),
                                nrows=fraction[1])
        time = pd.to_datetime(self.df.Date)
        self.df.Date = time
        self.window_size = window_size
        self.price_dependent = ['Open', 'High', 'Low', 'Close']

        # Indicators
        self.df, n, k = get_indicators(self.df)
        self.price_dependent = np.concatenate([self.price_dependent,k], axis=0)
        self.df = self.df.iloc[n:,:]
        self.df = self.df.reset_index(drop=True)

        # Adding time ints
        self.df.insert(loc=1, column='Minute', value=time.dt.minute.astype(np.int8))
        self.df.insert(loc=1, column='Hour', value=time.dt.hour.astype(np.int8))
        self.df.insert(loc=1, column='Day', value=time.dt.dayofweek.astype(np.int8))
        self.df.insert(loc=1, column='Month', value=time.dt.month.astype(np.int8))

        # Get all price dependent columns
        self.price_dependent = np.array([col in self.price_dependent for col in self.df.columns.values])

        # Group by day
        self.days = [g  for n, g in self.df.groupby(pd.Grouper(key='Date', freq='D')) if not len(g) < 60]
        days = []
        for day in self.days:
            index = day.index[0]
            day = pd.concat([self.df.iloc[index-self.window_size:index, :], day], axis=0)
            days.append(day)

        self.days = days
        self.num = len(self.days)

        print(f'\nData Generator initalized! Days available: {self.num}\n')


    def get_sample(self, k):
        num = np.random.choice(len(self.days), k)
        return self.days[num[0]]


def get_indicators(dataset):

    ## Price independent
    dataset['20sd'] = dataset['Close'].rolling(window=20).std()

    ## Price dependent
    # Create 7 and 21 days Moving Average
    dataset['ma9_pdt'] = dataset['Close'].rolling(window=15).mean()
    dataset['ma21_pdt'] = dataset['Close'].rolling(window=45).mean()
    
    # Create MACD
    dataset['26ema_pdt'] = dataset['Close'].ewm(span=26).mean()
    dataset['12ema_pdt'] = dataset['Close'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema_pdt']-dataset['26ema_pdt'])

    # Create Bollinger Bands
    dataset['BB_pdt']= dataset['Close'].rolling(window=20).mean()
    dataset['upper_band_pdt'] = dataset['BB_pdt'] + (dataset['20sd']*2)
    dataset['lower_band_pdt'] = dataset['BB_pdt'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema_pdt'] = dataset['Close'].ewm(com=0.5).mean()

    
    n = dataset.isnull().sum().max()
    k = [col for col in dataset.columns if '_pdt' in col]

    return dataset, n, k

if __name__ == "__main__":

    symbol = "SP500_M1"
    gen = DataGenerator(symbol=symbol, fraction=[int(1e6), 10000], window_size=10)

    # print(f' \n\nData types: {gen.df.dtypes} \nDataFrame: \n{gen.df.tail()}')
    print(f'Sample days: {len(gen.days)}')
