import pandas as pd
import polars as pl
from indicators import get_indicators, get_returns
from indicator_settings import settings

if __name__ == "__main__":
    storage = "D:/TS/"
    symbol = "SP500_D1"
    df = pl.read_csv(storage + symbol + ".csv").to_pandas()
    df.columns = ['date', 'open', 'high', 'low', 'close', '_', 'volume']
    df.pop("_")
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)
    print(df)
    input("s")
    df, flag = get_indicators(df=df, settings=settings, normalize=False, verbose=True)
    df = get_returns(df=df, historic_returns=[1,3,5,7,14,21,30], forward_returns=None)
    print(df)
    input("s")
    pl.from_pandas(df.reset_index()).to_csv(storage + symbol + "_TA.csv")