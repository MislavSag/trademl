import glob
import os
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mlfinlab.structural_breaks import (
    get_chu_stinchcombe_white_statistics,
    get_chow_type_stat, get_sadf)
import mlfinlab as ml
import trademl as tml


### GLOBAL

DATA_PATH = 'D:/market_data/usa/'


### PANDAS OPTIONS

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


### IMPORT DATA

# import data from mysql database and 
q = 'SELECT date, open, high, low, close, volume FROM SPY'
data = tml.modeling.utils.query_to_db(q, 'odvjet12_market_data_usa')
data.set_index(data.date, inplace=True)
data.drop(columns=['date'], inplace=True)
data.sort_index(inplace=True)

# remove big outliers
outlier_remove = tml.modeling.pipelines.OutlierStdRemove(50)
data_test = outlier_remove.fit_transform(data)

# NON SPY OLD WAY
# paths = glob.glob(DATA_PATH + 'ohlcv/*')
# contracts = [os.path.basename(p).replace('.h5', '') for p in paths]
# with pd.HDFStore(paths[0]) as store:
#     data = store.get(contracts[0])

    
# ADD FEATURES

# add technical indicators
periods = [5, 30, 60, 150, 300, 480, 2400, 12000]
data = tml.modeling.features.add_technical_indicators(data, periods=periods)
data.columns = [cl[0] if isinstance(cl, tuple) else cl for cl in data.columns]

# add ohlc transformations
data['high_low'] = data['high'] - data['low']
data['close_open'] = data['close'] - data['open']
data['close'].cummax()


# simple momentum
data['mom1'] = data['close'].pct_change(periods=1)
data['mom2'] = data['close'].pct_change(periods=2)
data['mom3'] = data['close'].pct_change(periods=3)
data['mom4'] = data['close'].pct_change(periods=4)
data['mom5'] = data['close'].pct_change(periods=5)

# Volatility
data['volatility_60'] = np.log(data['close']).diff().rolling(
    window=60, min_periods=60, center=False).std()
data['volatility_30'] = np.log(data['close']).diff().rolling(
    window=30, min_periods=30, center=False).std()
data['volatility_15'] = np.log(data['close']).diff().rolling(
    window=15, min_periods=15, center=False).std()
data['volatility_10'] = np.log(data['close']).diff().rolling(
    window=10, min_periods=10, center=False).std()
data['volatility_5'] =np.log(data['close']).diff().rolling(
    window=5, min_periods=5, center=False).std()

# Serial Correlation (Takes time) TO SLOW
# window_autocorr = 50

# data['autocorr_1'] = np.log(data['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=1), raw=False)
# data['autocorr_2'] = np.log(data['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=2), raw=False)
# data['autocorr_3'] = np.log(data['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=3), raw=False)
# data['autocorr_4'] = np.log(data['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=4), raw=False)
# data['autocorr_5'] = np.log(data['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=5), raw=False)

# Skewness
data['skew_60'] = np.log(data['close']).diff().rolling(
    window=60, min_periods=60, center=False).skew()
data['skew_30'] = np.log(data['close']).diff().rolling(
    window=30, min_periods=30, center=False).skew()
data['skew_15'] = np.log(data['close']).diff().rolling(
    window=15, min_periods=15, center=False).skew()
data['skew_10'] = np.log(data['close']).diff().rolling(
    window=10, min_periods=10, center=False).skew()
data['skew_5'] =np.log(data['close']).diff().rolling(
    window=5, min_periods=5, center=False).skew()

# kurtosis
data['kurtosis_60'] = np.log(data['close']).diff().rolling(
    window=60, min_periods=60, center=False).kurt()
data['kurtosis_30'] = np.log(data['close']).diff().rolling(
    window=30, min_periods=30, center=False).kurt()
data['kurtosis_15'] = np.log(data['close']).diff().rolling(
    window=15, min_periods=15, center=False).kurt()
data['kurtosis_10'] = np.log(data['close']).diff().rolling(
    window=10, min_periods=10, center=False).kurt()
data['kurtosis_5'] =np.log(data['close']).diff().rolling(
    window=5, min_periods=5, center=False).kurt()

# remove na
data.isna().sum().sort_values(ascending=False).head(20)
data.drop(columns=['T396000'], inplace=True)  # T396000 575994, TRIX96000 TEMA96000 ADXR96000 287998, ADX96000 DEMA96000 191998
# T312000 71994, ADXR12000 TRIX12000  35998, TEMA12000 35997, ADX12000 APO_5 23999, DEMA12000 23998, T32400 14394
# data.drop(columns=['T312000', 'ADXR12000', 'TRIX12000', 'TEMA12000', 'ADX12000', 'APO_5', 
#                    'DEMA12000', 'T32400'], inplace=True)
cols_remove_na = range((np.where(data.columns == 'volume')[0].item() + 1), data.shape[1])
data.dropna(subset=data.columns[cols_remove_na], inplace=True)


###  STATIONARITY
ohlc = data[['open', 'high', 'low', 'close']]  # save for later
ohlc.columns = ['open_orig', 'high_orig', 'low_orig', 'close_orig']
# get dmin for every column
stationaryCols, min_d = min_ffd_all_cols(data)

# save to github for later 
min_dmin_d_save_for_backtesting = pd.Series(0, index=data.columns)
min_dmin_d_save_for_backtesting.update(min_d)
min_dmin_d_save_for_backtesting.dropna(inplace=True)
min_dmin_d_save_for_backtesting.to_csv('min_d.csv', sep=';')

# convert unstationary to stationary
data = unstat_cols_to_stat(data, min_d, stationaryCols)  # tml.modeling.stationarity.unstat_cols_to_stat
data.dropna(inplace=True)

# merge orig ohlc to spyStat
data = data.merge(ohlc, how='left', left_index=True, right_index=True)
data.head()
data.tail()


### STRUCTURAL BRAKES

@njit
def _get_dfc_for_t(molecule_range, series_lag_values_start, series_diff, series_lag_values):
    """
    Get Chow-Type Dickey-Fuller Test statistics for each index in molecule
    :param molecule_range: (np.array) of dates to test
    :param series_lag_values_start: (int) offset series because of min_length
    :return: (pd.Series) fo statistics for each index from molecule
    """
    dfc_series = []
    for i in molecule_range:
        ### TEST
        # index = molecule[0]
        ### TEST
        series_lag_values_ = series_lag_values.copy()
        series_lag_values_[:(series_lag_values_start + i)] = 0  # D_t* indicator: before t* D_t* = 0

        # define x and y for regression
        y = series_diff
        x = series_lag_values_.reshape(-1, 1)
        
        # Get regression coefficients estimates
        xy = x.transpose() @ y
        xx = x.transpose() @ x

        # calculate to check for singularity
        det = np.linalg.det(xx)

        # get coefficient and std from linear regression
        if det == 0:
            b_mean = [np.nan]
            b_std = [[np.nan, np.nan]]
        else:
            xx_inv = np.linalg.inv(xx)
            coefs = xx_inv @ xy
            err = y - (x @ coefs)
            coef_vars = np.dot(np.transpose(err), err) / (x.shape[0] - x.shape[1]) * xx_inv
            
        b_estimate, b_var = coefs[0], coef_vars[0][0]
        dfc_series.append(b_estimate / (b_var ** 0.5))
        
    return dfc_series


def get_chow_type_stat(series: pd.Series, min_length: int = 20) -> pd.Series:
    """
    Multithread implementation of Chow-Type Dickey-Fuller Test, p.251-252
    :param series: (pd.Series) series to test
    :param min_length: (int) minimum sample length used to estimate statistics
    :param num_threads: (int): number of cores to use
    :return: (pd.Series) of Chow-Type Dickey-Fuller Test statistics
    """
    # Indices to test. We drop min_length first and last values
    molecule = series.index[min_length:series.shape[0] - min_length]
    molecule = molecule.values
    molecule_range = np.arange(0, len(molecule))

    series_diff = series.diff().dropna()
    series_diff = series_diff.values
    series_lag = series.shift(1).dropna()
    series_lag_values = series_lag.values
    series_lag_times_ = series_lag.index.values
    series_lag_values_start = np.where(series_lag_times_ == molecule[0])[0].item() + 1
    
    dfc_series = _get_dfc_for_t(molecule_range, series_lag_values_start, series_diff, series_lag_values)
    
    dfc_series = pd.Series(dfc_series, index=molecule)
    
    return dfc_series


# chow = get_chow_type_stat(series=spy_with_vix['close_orig'], min_length=20)


### SAVE SPY WITH VIX

# save SPY
save_path = DATA_PATH + 'ohlcv_features/' + 'SPY' + '.h5'
with pd.HDFStore(save_path) as store:
    store.put('SPY', data)
