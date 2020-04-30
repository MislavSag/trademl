# fundamental modules
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import pyplot
# my functions
# from features import add_technical_indicators
# from stationarity import min_ffd_plot, min_ffd_value, unstat_cols_to_stat
from mlfinlab.structural_breaks import (
    get_chu_stinchcombe_white_statistics,
    get_chow_type_stat, get_sadf)
import mlfinlab as ml
import trademl as tml


### GLOBAL (CONFIGS)

DATA_PATH = 'C:/Users/Mislav/algoAItrader/data/'


### PANDAS OPTIONS

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


### IMPORT DATA

spy = Path(DATA_PATH + 'spy.h5')
with pd.HDFStore(spy) as store:
    spy = store.get('spy')
spy.drop(columns=['average', 'barCount', 'vixAverage', 'vixBarCount'],
         inplace=True)  # from IB, remove for now

    
# ADD FEATURES

# add technical indicators
periods = [5, 30, 60, 300, 480, 2400, 12000, 96000]
spy = tml.modeling.features.add_technical_indicators(spy, periods=periods)
spy.columns = [cl[0] if isinstance(cl, tuple) else cl for cl in spy.columns]
spy.isna().sum().sort_values()
spy.drop(columns=['T396000'], inplace=True)

# add ohlc transformations
spy['high_low'] = spy['high'] - spy['low']
spy['close_open'] = spy['close'] - spy['open']

# simple momentum
spy['mom1'] = spy['close'].pct_change(periods=1)
spy['mom2'] = spy['close'].pct_change(periods=2)
spy['mom3'] = spy['close'].pct_change(periods=3)
spy['mom4'] = spy['close'].pct_change(periods=4)
spy['mom5'] = spy['close'].pct_change(periods=5)

# Volatility
spy['volatility_60'] = np.log(spy['close']).diff().rolling(
    window=60, min_periods=60, center=False).std()
spy['volatility_30'] = np.log(spy['close']).diff().rolling(
    window=30, min_periods=30, center=False).std()
spy['volatility_15'] = np.log(spy['close']).diff().rolling(
    window=15, min_periods=15, center=False).std()
spy['volatility_10'] = np.log(spy['close']).diff().rolling(
    window=10, min_periods=10, center=False).std()
spy['volatility_5'] =np.log(spy['close']).diff().rolling(
    window=5, min_periods=5, center=False).std()

# Serial Correlation (Takes time)
window_autocorr = 50

spy['autocorr_1'] = np.log(spy['close']).diff().rolling(
    window=window_autocorr, min_periods=window_autocorr,
    center=False).apply(lambda x: x.autocorr(lag=1), raw=False)
spy['autocorr_2'] = np.log(spy['close']).diff().rolling(
    window=window_autocorr, min_periods=window_autocorr,
    center=False).apply(lambda x: x.autocorr(lag=2), raw=False)
spy['autocorr_3'] = np.log(spy['close']).diff().rolling(
    window=window_autocorr, min_periods=window_autocorr,
    center=False).apply(lambda x: x.autocorr(lag=3), raw=False)
spy['autocorr_4'] = np.log(spy['close']).diff().rolling(
    window=window_autocorr, min_periods=window_autocorr,
    center=False).apply(lambda x: x.autocorr(lag=4), raw=False)
spy['autocorr_5'] = np.log(spy['close']).diff().rolling(
    window=window_autocorr, min_periods=window_autocorr,
    center=False).apply(lambda x: x.autocorr(lag=5), raw=False)

# Skewness
spy['skew_60'] = np.log(spy['close']).diff().rolling(
    window=60, min_periods=60, center=False).skew()
spy['skew_30'] = np.log(spy['close']).diff().rolling(
    window=30, min_periods=30, center=False).skew()
spy['skew_15'] = np.log(spy['close']).diff().rolling(
    window=15, min_periods=15, center=False).skew()
spy['skew_10'] = np.log(spy['close']).diff().rolling(
    window=10, min_periods=10, center=False).skew()
spy['skew_5'] =np.log(spy['close']).diff().rolling(
    window=5, min_periods=5, center=False).skew()

# kurtosis
spy['kurtosis_60'] = np.log(spy['close']).diff().rolling(
    window=60, min_periods=60, center=False).kurt()
spy['kurtosis_30'] = np.log(spy['close']).diff().rolling(
    window=30, min_periods=30, center=False).kurt()
spy['kurtosis_15'] = np.log(spy['close']).diff().rolling(
    window=15, min_periods=15, center=False).kurt()
spy['kurtosis_10'] = np.log(spy['close']).diff().rolling(
    window=10, min_periods=10, center=False).kurt()
spy['kurtosis_5'] =np.log(spy['close']).diff().rolling(
    window=5, min_periods=5, center=False).kurt()

# remove na
spy.isna().sum().sort_values(ascending=False).head(20)
cols_remove_na = range((np.where(spy.columns == 'vixVolume')[0].item() + 1), spy.shape[1])
spy.dropna(subset=spy.columns[cols_remove_na], inplace=True)


###  STATIONARITY

spy_with_vix = tml.modeling.stationarity.unstat_cols_to_stat(spy)
spy_with_vix.dropna(inplace=True)


# merge orig ohlc to spyStat
ohlc = spy[['open', 'high', 'low', 'close']]
ohlc.columns = ['open_orig', 'high_orig', 'low_orig', 'close_orig']
spy_with_vix = spy_with_vix.merge(ohlc, how='left', left_index=True, right_index=True)
print(spy_with_vix.shape)
display(spy_with_vix.head())
display(spy_with_vix.tail())


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


chow = get_chow_type_stat(series=spy_with_vix['close_orig'], min_length=20)


### SAVE SPY WITH VIX

# save SPY
spy_with_vix_path = DATA_PATH + '/spy_with_vix.h5'
with pd.HDFStore(spy_with_vix_path) as store:
    store.put(' ', spy_with_vix)
