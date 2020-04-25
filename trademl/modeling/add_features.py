# # fundamental modules
# import numpy as np
# import pandas as pd
# from pathlib import Path
# import matplotlib.pyplot as plt
# from matplotlib import pyplot
# # my functions
# # from features import add_technical_indicators
# # from stationarity import min_ffd_plot, min_ffd_value, unstat_cols_to_stat
# from mlfinlab.structural_breaks import (
#     get_chu_stinchcombe_white_statistics,
#     get_chow_type_stat, get_sadf)
# import trademl as tml


# ### GLOBAL (CONFIGS)

# DATA_PATH = 'C:/Users/Mislav/algoAItrader/data/'


# ### PANDAS OPTIONS

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


# ### IMPORT DATA

# spy = Path(DATA_PATH + 'spy.h5')
# with pd.HDFStore(spy) as store:
#     spy = store.get('spy')
# spy.drop(columns=['average', 'barCount', 'vixAverage', 'vixBarCount'],
#          inplace=True)  # from IB, remove for now

    
# # ADD FEATURES

# # add technical indicators
# periods = [5, 30, 60, 300, 480, 2400, 12000, 96000]
# spy = tml.modeling.features.add_technical_indicators(spy, periods=periods)
# spy.columns = [cl[0] if isinstance(cl, tuple) else cl for cl in spy.columns]
# spy.isna().sum().sort_values()
# spy.drop(columns=['T396000'], inplace=True)

# # add ohlc transformations
# spy['high_low'] = spy['high'] - spy['low']
# spy['close_open'] = spy['close'] - spy['open']

# # simple momentum
# spy['mom1'] = spy['close'].pct_change(periods=1)
# spy['mom2'] = spy['close'].pct_change(periods=2)
# spy['mom3'] = spy['close'].pct_change(periods=3)
# spy['mom4'] = spy['close'].pct_change(periods=4)
# spy['mom5'] = spy['close'].pct_change(periods=5)

# # Volatility
# spy['volatility_60'] = np.log(spy['close']).diff().rolling(
#     window=60, min_periods=60, center=False).std()
# spy['volatility_30'] = np.log(spy['close']).diff().rolling(
#     window=30, min_periods=30, center=False).std()
# spy['volatility_15'] = np.log(spy['close']).diff().rolling(
#     window=15, min_periods=15, center=False).std()
# spy['volatility_10'] = np.log(spy['close']).diff().rolling(
#     window=10, min_periods=10, center=False).std()
# spy['volatility_5'] =np.log(spy['close']).diff().rolling(
#     window=5, min_periods=5, center=False).std()

# # Serial Correlation (Takes time)
# window_autocorr = 50

# spy['autocorr_1'] = np.log(spy['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=1), raw=False)
# spy['autocorr_2'] = np.log(spy['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=2), raw=False)
# spy['autocorr_3'] = np.log(spy['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=3), raw=False)
# spy['autocorr_4'] = np.log(spy['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=4), raw=False)
# spy['autocorr_5'] = np.log(spy['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=5), raw=False)

# # Skewness
# spy['skew_60'] = np.log(spy['close']).diff().rolling(
#     window=60, min_periods=60, center=False).skew()
# spy['skew_30'] = np.log(spy['close']).diff().rolling(
#     window=30, min_periods=30, center=False).skew()
# spy['skew_15'] = np.log(spy['close']).diff().rolling(
#     window=15, min_periods=15, center=False).skew()
# spy['skew_10'] = np.log(spy['close']).diff().rolling(
#     window=10, min_periods=10, center=False).skew()
# spy['skew_5'] =np.log(spy['close']).diff().rolling(
#     window=5, min_periods=5, center=False).skew()

# # kurtosis
# spy['kurtosis_60'] = np.log(spy['close']).diff().rolling(
#     window=60, min_periods=60, center=False).kurt()
# spy['kurtosis_30'] = np.log(spy['close']).diff().rolling(
#     window=30, min_periods=30, center=False).kurt()
# spy['kurtosis_15'] = np.log(spy['close']).diff().rolling(
#     window=15, min_periods=15, center=False).kurt()
# spy['kurtosis_10'] = np.log(spy['close']).diff().rolling(
#     window=10, min_periods=10, center=False).kurt()
# spy['kurtosis_5'] =np.log(spy['close']).diff().rolling(
#     window=5, min_periods=5, center=False).kurt()

# # remove na
# spy.isna().sum().sort_values(ascending=False).head(20)
# cols_remove_na = range((np.where(spy.columns == 'vixVolume')[0].item() + 1), spy.shape[1])
# spy.dropna(subset=spy.columns[cols_remove_na], inplace=True)




# ############ PROVJERITI JE LI FUNKCIJA IZ TRADEMLA ISPRAVNA!!! ##########

# data = spy.copy()
# # stationarity tests
# adfTest = data.apply(lambda x: adfuller(x, maxlag=1, regression='c',
#                                         autolag=None), axis=0)
# adfTestPval = [adf[1] for adf in adfTest]
# adfTestPval = pd.Series(adfTestPval)
# stationaryCols = data.loc[:, (adfTestPval > 0.1).to_list()].columns

# min_d = data[stationaryCols[57:59]].apply(lambda x: min_ffd_value(x.to_frame(), seq))

# # make stationary spy
# dataStationary = data[stationaryCols].loc[:, min_d > 0]
# diff_amt_args = min_d[min_d > 0].to_list()
# for i, col in enumerate(dataStationary.columns):
#     print("Making ", col, " stationary")
#     dataStationary[col] = frac_diff_ffd(dataStationary[col].values, diff_amt_args[i])
    

# ############ PROVJERITI JE LI FUNKCIJA IZ TRADEMLA ISPRAVNA!!! ##########


# ###  STATIONARITY

# spy_with_vix = unstat_cols_to_stat(spy)
# spy_with_vix.dropna(inplace=True)


# # merge close to spyStat
# spy_with_vix = spy_with_vix.merge(close_orig, how='left', left_index=True, right_index=True)
# print(spy_with_vix.shape)
# display(spy_with_vix.head())
# display(spy_with_vix.tail())


# ### SAVE SPY WITH VIX

# # save SPY
# spy_with_vix_path = DATA_PATH + '/spy_with_vix.h5'
# with pd.HDFStore(spy_with_vix_path) as store:
#     store.put('spy_with_vix', spy_with_vix)
