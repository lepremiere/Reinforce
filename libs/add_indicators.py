import pandas as pd
import pandas_ta as ta 

if __name__ == "__main__":
    storage = "D:/Data/"
    symbol = "SP500_M5"
    df = pd.read_csv(storage + symbol + ".csv")
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df.set_index(pd.DatetimeIndex(df["date"]), inplace=True)
    df.pop('date')

    df.ta.strategy()

    with open('./2_storage/{symbol}.txt', 'w') as f:
        for i in range(len(df.dtypes)):
            f.write(f'{df.columns[i]}:, dtype: {df.dtypes[i]}, NaNs: {df[df.columns[i]].isnull().sum()}\n')

    df.to_csv(storage + symbol + "_TA.csv")