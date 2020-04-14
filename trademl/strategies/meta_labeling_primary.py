import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import talib


def vix_change_strategy(data, vix_close_column='vixClose', vix_window_size=3):
    """
    Trading strategy that buy stock if VIX is grater than VIX rolling mean.
    
    :param data: (pd.DataFrame) pandas DF with VIX close as on of columns.
    :param vix_close_column: (flostrat) name of VIX close column.
    :param vix_window_size: (int) rolling window.
    :return data: (pd.DataFrame) Data frame with sign column
    """
    # calculate VIX close MA
    vixMA = data[vix_close_column].rolling(window=vix_window_size).mean()
    vixMA.dropna(inplace=True)  #remove NA values we get in prevous step
    
    # buy if VIX increases
    data = data.iloc[(vix_window_size-1):, :]  #we have to merge with vixMA
    data['side'] = np.where(data[vix_close_column] > vixMA, -1, 1)
    # data = data.iloc[1:, :]  # PANDAS DOESN'T USE THREE-VALUED LOGIC!!!

    # Remove Look ahead biase by lagging the signal
    data['side'] = data['side'].shift(1)
    data.dropna(inplace=True)
    
    return data


def crossover_strategy(data, close_column, fast_ma=25, slow_ma=50):
    """
    Trading strategy that buy stock if VIX is grater than VIX rolling mean.
    
    :param data: (pd.DataFrame) pandas DF with close price as on of columns.
    :param close_column: (flostrat) name of close price column.
    :param fast_ma: (int) fast SMA.
    :param slow_ma: (int) slow SMA.
    :return data: (pd.DataFrame) Data frame with sign column
    """
    # calculate SMA
    sma_fast = pd.Series(talib.SMA(data[close_column], fast_ma))
    sma_slow = pd.Series(talib.SMA(data[close_column], slow_ma))
    
    # Primary model: make signs
    data = data.iloc[(slow_ma-1):, :]  #we have to merge with vixM
    data.loc[(sma_fast >= sma_slow), 'side'] = 1
    data.loc[(sma_fast < sma_slow), 'side'] = -1
    
    # Remove Look ahead biase by lagging the signal
    data['side'] = data['side'].shift(1)
    data.dropna(inplace=True)
    
    return data


def bbands_Strategy(data, close_column, period=5, nbdevup_=2, nbdevdn_=2, matype_=0):
    """
    Trading strategy that buy stock if VIX is grater than VIX rolling mean.
    
    :param data: (pd.DataFrame) pandas DF with close price as on of columns.
    :param close_column: (flostrat) name of close price column.
    :return data: (pd.DataFrame) Data frame with sign column
    """
    # calculate bbands
    upperband, middleband, lowerband = talib.BBANDS(data[close_column],
                                                    timeperiod=period,
                                                    nbdevup=nbdevup_,
                                                    nbdevdn=nbdevdn_,
                                                    matype=matype_)
    lowerband = pd.Series(lowerband)
    upperband = pd.Series(upperband)
    
    # Primary model: make signs
    data.loc[(data[close_column] <= lowerband), 'side'] = 1
    data.loc[(data[close_column] >= upperband), 'side'] = -1
    data = data.iloc[(period-1):, :]
    
    # Remove Look ahead biase by lagging the signal
    data['side'] = data['side'].shift(1)
    data.dropna(inplace=True)
    
    return data
