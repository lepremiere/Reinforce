import pandas as pd
import pandas_ta as ta 

if __name__ == "__main__":
    df = pd.read_csv("D:/Data/SP500_M1.csv")
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df.set_index(pd.DatetimeIndex(df["date"]), inplace=True)
    df.pop('date')

    df.ta.strategy()

    with open('indicators.txt', 'w') as f:
        for i in range(len(df.dtypes)):
            f.write(f'{df.columns[i]}:, dtype: {df.dtypes[i]}, NaNs: {df[df.columns[i]].isnull().sum()}\n')

    df.to_csv('D:/Data/SP500_M1_TA.csv')