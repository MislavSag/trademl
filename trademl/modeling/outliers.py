import pandas as pd
import numpy as np


def remove_ohlc_ouliers(data, threshold_up=1, threshold_down=-0.3):
    """
    Remove ohlc obervations where value grater than threshold.
    That is, remove rows where growth grater than threshold.

    Returns:
        pd.DataFrame
    """
    ohlcCols = ["open", 'high', 'low', 'close']
    data = data.loc[
        (data[ohlcCols].pct_change(1, freq='Min').fillna(0) < threshold_up).all(1) &
        (data[ohlcCols].pct_change(1, freq='Min').shift(-1).fillna(0) > threshold_down).all(1)
    ]
    data = data.loc[
        (data[ohlcCols].pct_change(1, freq='Min').fillna(0) > threshold_down).all(1) &
        (data[ohlcCols].pct_change(1, freq='Min').shift(-1).fillna(0) < threshold_up).all(1)
    ]

    return data


def remove_ourlier_diff_median(data, median_scaler=25):
    """
    Remove outliers by removing observations where differene is grater than
    daiy difference times the median scaler.

    :param data: (pd.DataFrame) with ohlc data
    :return: (pd.DataFrame) with removed outliers
    """
    daily_diff = (data.resample('D').last().dropna().diff() + 0.005) * median_scaler
    daily_diff['diff_date'] = daily_diff.index.strftime('%Y-%m-%d')
    data_test = data.diff()
    data_test['diff_date'] = data_test.index.strftime('%Y-%m-%d')
    data_test_diff = pd.merge(data_test, daily_diff, on='diff_date')
    indexer = ((np.abs(data_test_diff['close_x']) < np.abs(data_test_diff['close_y'])) &
            (np.abs(data_test_diff['open_x']) < np.abs(data_test_diff['open_y'])) &
            (np.abs(data_test_diff['high_x']) < np.abs(data_test_diff['high_y'])) & 
            (np.abs(data_test_diff['low_x']) < np.abs(data_test_diff['low_y'])))
    # indexer = (indexer | data_test_diff['close_y'].isna())
    data_final = data.loc[indexer.values, :]
    
    return data_final
