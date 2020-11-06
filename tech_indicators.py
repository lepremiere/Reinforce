def get_technical_indicators(dataset):

    # Create 7 and 21 days Moving Average
    dataset['ma9'] = dataset['Close'].rolling(window=15).mean()
    dataset['ma21'] = dataset['Close'].rolling(window=45).mean()
    
    # Create MACD
    dataset['26ema'] = dataset['Close'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Close'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    # Create Bollinger Bands
    dataset['20sd'] = dataset['Close'].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema'] = dataset['Close'].ewm(com=0.5).mean()
    
    # Create Momentum
    dataset['momentum'] = dataset['Close']-1

    i = dataset.isnull().sum().max()
    dataset = dataset.iloc[i:]
    return dataset, i