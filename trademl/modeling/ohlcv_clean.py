import os
import glob
import numpy as np
import pandas as pd
import numba
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mlfinlab.structural_breaks import (
    get_chu_stinchcombe_white_statistics,
    get_chow_type_stat, get_sadf)
import mlfinlab as ml
import mlfinlab.microstructural_features as micro
import trademl as tml
from trademl.modeling.utils import time_method



### HYPERPARAMETERS
save_path = 'D:/market_data/usa/ohlcv_features'
add_ta = False
ta_periods = [5, 30, 480, 960, 2400, 4800, 9600]
add_labels = False
env_directory = None  # os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
median_outlier_thrteshold = 20


### IMPORT DATA
# import data from mysql database and 
contract = 'SPY_clean'
q = 'SELECT date, open, high, low, close, volume FROM SPY'
data = tml.modeling.utils.query_to_db(q, 'odvjet12_market_data_usa')
data.set_index(data.date, inplace=True)
data.drop(columns=['date'], inplace=True)
security = data.sort_index()




daily_diff = (security.resample('D').last().dropna().diff().abs() + 0.1) * 20
daily_diff['diff_date'] = daily_diff.index.strftime('%Y-%m-%d')
data_test = security.diff()
data_test['diff_date'] = data_test.index.strftime('%Y-%m-%d')
data_test_diff = pd.merge(data_test, daily_diff, on='diff_date')
indexer = ((np.abs(data_test_diff['close_x']) < np.abs(data_test_diff['close_y'])) &
        (np.abs(data_test_diff['open_x']) < np.abs(data_test_diff['open_y'])) &
        (np.abs(data_test_diff['high_x']) < np.abs(data_test_diff['high_y'])) & 
        (np.abs(data_test_diff['low_x']) < np.abs(data_test_diff['low_y'])))
np.where(indexer == False)
data_test_diff.loc[np.where(indexer == False)]
data_final = data.loc[indexer.values, :]
data_final['high'].plot()

# indexer = (indexer | data_test_diff['close_y'].isna())

### REMOVE OUTLIERS
security = tml.modeling.outliers.remove_ourlier_diff_median(security, median_outlier_thrteshold)

security = tml.modeling.outliers.remove_ohlc_ouliers(data, threshold_up=0.5, threshold_down=-0.3)
security = tml.modeling.outliers.remove_ourlier_diff_median(security, median_outlier_thrteshold)

########### TEST FOR INDICATORS ##############
# data_sample = data.iloc[:20000]
# periods = [5, 30, 60, 480, 960, 2400, 4800, 9600]
# data_sample = tml.modeling.features.add_technical_indicators(data_sample, periods=periods)
# data_sample.columns = [cl[0] if isinstance(cl, tuple) else cl for cl in data_sample.columns]
# data_sample.isna().sum().sort_values()
########### TEST FOR INDICATORS ##############


### 1) ADD FEATURES
# add technical indicators
if add_ta:
    # periods = [5, 30, 480, 960, 2400, 4800, 9600]
    data = tml.modeling.features.add_technical_indicators(data, periods=ta_periods)
    data.columns = [cl[0] if isinstance(cl, tuple) else cl for cl in data.columns]


def add_ohlcv_features(data):
    """
    Calculate features based on OHLCV data and add tham to data feame.
    """

    # add ohlc transformations
    data['high_low'] = data['high'] - data['low']
    data['close_open'] = data['close'] - data['open']
    data['close_ath'] = data['close'].cummax()
        
    # simple momentum
    data['momentum1'] = data['close'].pct_change(periods=1)
    data['momentum2'] = data['close'].pct_change(periods=2)
    data['momentum3'] = data['close'].pct_change(periods=3)
    data['momentum4'] = data['close'].pct_change(periods=4)
    data['momentum5'] = data['close'].pct_change(periods=5)
    
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
    
    # microstructural features
    data['roll_measure'] = micro.get_roll_measure(data['close'])
    data['corwin_schultz_est'] = micro.get_corwin_schultz_estimator(
        data['high'], data['low'], 100)
    data['bekker_parkinson_vol'] = micro.get_bekker_parkinson_vol(
        data['high'], data['low'], 100)
    data['kyle_lambda'] = micro.get_bekker_parkinson_vol(
        data['close'], data['volume'])
    data['amihud_lambda'] = micro.get_bar_based_amihud_lambda(
        data['close'], data['volume'])
    data['hasbrouck_lambda'] = micro.get_bar_based_hasbrouck_lambda(
        data['close'], data['volume'])
    tick_diff = data['close'].diff()
    data['tick_rule'] = np.where(tick_diff != 0,
                                np.sign(tick_diff),
                                np.sign(tick_diff).shift(periods=-1))
    
    ### ADD VIX TO DATABASE
    q = 'SELECT date, open AS open_vix, high AS high_vix, low AS low_vix, \
        close AS close_vix, volume AS volume_vix FROM VIX'
    data_vix = tml.modeling.utils.query_to_db(q, 'odvjet12_market_data_usa')
    data_vix.set_index(data_vix.date, inplace=True)
    data_vix.drop(columns=['date'], inplace=True)
    data_vix.sort_index(inplace=True)
    # merge spy and vix with merge_asof which uses nearest back value for NA
    data_vix = data_vix.sort_index()
    data = pd.merge_asof(data, data_vix, left_index=True, right_index=True)
    
    ### VIX FEATURES
    data['vix_high_low'] = data['high'] - data['low']
    data['vix_close_open'] = data['close'] - data['open']

    
    return data


data = add_ohlcv_features(data)


### REMOVE NAN
data.isna().sum().sort_values(ascending=False).head(60)
if add_ta:
    data = data.loc[:, data.isna().sum() < (max(periods) + 100)]
cols_remove_na = range((np.where(data.columns == 'volume')[0].item() + 1), data.shape[1])
data.dropna(subset=data.columns[cols_remove_na], inplace=True)


### 2) LABELING (COMPUTATIONALLY INTENSIVE)
if add_labels:
    # trend scanning
    def add_trend_scanning_label(data, look_forward, col_prefix=''):
        ts_1_day = tml.modeling.pipelines.trend_scanning_labels(
            data['close'], t_events=data.index, look_forward_window=observatins_per_day,
            min_sample_length=30, step=2
        )
        ts_1_day = ts_1_day.add_prefix(col_prefix)
        return pd.concat([data, ts_1_day], axis=1)

    
    observatins_per_day = int(pd.value_counts(data.index.normalize(), sort=False).mean())
    data = add_trend_scanning_label(data, observatins_per_day, 'day_1_')
    data = add_trend_scanning_label(data, observatins_per_day*2, 'day_2_')
    data = add_trend_scanning_label(data, observatins_per_day*3, 'day_3_')
    data = add_trend_scanning_label(data, observatins_per_day*5, 'day_5_')
    data = add_trend_scanning_label(data, observatins_per_day*10, 'day_10_')
    data = add_trend_scanning_label(data, observatins_per_day*20, 'day_20_')
    data = add_trend_scanning_label(data, observatins_per_day*30, 'day_30_')
    data = add_trend_scanning_label(data, observatins_per_day*60, 'day_60_')

    # triple-barrier labeling


### 3) STRUCTURAL BRAKES

# CHOW
close_weekly = data['close'].resample('W').last().dropna()
close_weekly_log = np.log(close_weekly)
chow = tml.modeling.structural_breaks.get_chow_type_stat(
    series=close_weekly_log, min_length=10)
breakdate = chow.loc[chow == chow.max()]
data['chow_segment'] = 0
data['chow_segment'][breakdate.index[0]:] = 1
data['chow_segment'].loc[breakdate.index[0]:] = 1
data['chow_segment'] = np.where(data.index < breakdate.index[0], 0, 1)
data['chow_segment'].value_counts()

# SADF
# sadf_linear =ml.structural_breaks.get_sadf(
#     close_weekly_log, min_length=20, add_const=True, model='linear', phi=0.5, num_threads=1, lags=5)
# sadf_quadratic = ml.structural_breaks.get_sadf(
#     close_weekly_log, min_length=20, add_const=True, model='quadratic', phi=0.5, num_threads=1, lags=5)
# sadf_poly_1 = ml.structural_breaks.get_sadf(
#     close_weekly_log, min_length=20, add_const=True, model='sm_poly_1', phi=0.5, num_threads=1, lags=5)
# sadf_poly_2 = ml.structural_breaks.get_sadf(
#     close_weekly_log, min_length=20, add_const=True, model='sm_poly_2', phi=0.5, num_threads=1, lags=5)
# sadf_power = ml.structural_breaks.get_sadf(
#     close_weekly_log, min_length=20, add_const=True, model='sm_power', phi=0.5, num_threads=1, lags=5)
# sadf = pd.concat([pd.Series(sadf_linear), pd.Series(sadf_quadratic), pd.Series(sadf_poly_1),
#                   pd.Series(sadf_poly_2), pd.Series(sadf_power)], axis=1)
# sadf.columns = ['sadf_linear', 'sadf_quadratic', 'sadf_poly_1', 'sadf_poly_2', 'sadf_power']
# sadf.loc[sadf['sadf_linear'] == sadf['sadf_linear'].max()]
# # pd.Series(sadf_linear).plot()
# # pd.Series(sadf_quadratic).plot()
# # pd.Series(sadf_poly_1).plot()
# # pd.Series(sadf_poly_2).plot()
# # pd.Series(sadf_power).plot()

### STATIONARITY
# save original ohlcv, I will need it later
ohlc = data[['open', 'high', 'low', 'close', 'chow_segment']]
ohlc.columns = ['open_orig', 'high_orig', 'low_orig', 'close_orig', 'chow_segment']
stationariti_test_cols = data.columns[:np.where(data.columns == 'vix_close_open')[0][0]]
stationaryCols, min_d = tml.modeling.stationarity.min_ffd_all_cols(data[stationariti_test_cols])



### STATIONARITY
# save original ohlcv, I will need it later
ohlc = data[['open', 'high', 'low', 'close', 'chow_segment']]
ohlc.columns = ['open_orig', 'high_orig', 'low_orig', 'close_orig', 'chow_segment']
stationariti_test_cols = data.columns[:np.where(data.columns == 'vix_close_open')[0][0]]
stationaryCols, min_d = tml.modeling.stationarity.min_ffd_all_cols(data[stationariti_test_cols])

# save to github for later 
min_dmin_d_save_for_backtesting = pd.Series(0, index=security.columns)
min_dmin_d_save_for_backtesting.update(min_d)
min_dmin_d_save_for_backtesting.dropna(inplace=True)
min_dmin_d_save_for_backtesting.to_csv(
    'C:/Users/Mislav/Documents/GitHub/trademl/data/min_d_' + contract + '.csv', sep=';')

# convert unstationary to stationary
security = tml.modeling.stationarity.unstat_cols_to_stat(security, min_d, stationaryCols)  # tml.modeling.stationarity.unstat_cols_to_stat
security = security.dropna()
security = security[stationaryCols].add_prefix('fracdiff_')

# merge orig_ohlc
security = security.merge(ohlc, how='left', left_index=True, right_index=True)



### SAVE
# save localy
file_name = 'SPY_raw'
if add_ta:
    file_name = file_name + '_ta'
if add_labels:
    file_name = file_name + '_labels'
save_path_local = os.path.join(Path(save_path), file_name + '.h5')
if os.path.exists(save_path_local):
    os.remove(save_path_local)
with pd.HDFStore(save_path_local) as store:
    store.put(file_name, data)
# save to mfiles
if env_directory is not None:
    mfiles_client = tml.modeling.utils.set_mfiles_client(env_directory)
    tml.modeling.utils.destroy_mfiles_object(mfiles_client, [file_name + '.h5'])
    wd = os.getcwd()
    os.chdir(Path(save_path))
    mfiles_client.upload_file(file_name, object_type='Dokument')
    os.chdir(wd)


###  STATIONARITY
# save original ohlcv, I will need it later
ohlc = data[['open', 'high', 'low', 'close', 'chow_segment']]
ohlc.columns = ['open_orig', 'high_orig', 'low_orig', 'close_orig', 'chow_segment']


from statsmodels.tsa.stattools import adfuller
adfTest = data.apply(lambda x: adfuller(x, 
                                        maxlag=1,
                                        regression='c',
                                        autolag=None),
                        axis=0)
adfTestPval = [adf[1] for adf in adfTest]
adfTestPval = pd.Series(adfTestPval)
stationaryCols = data.loc[:, (adfTestPval > 0.1).to_list()].columns

# get minimum values of d for every column
seq = np.linspace(0, 1, 16)
min_d = data[stationaryCols].apply(lambda x: min_ffd_value(x.to_frame(), seq))






# get dmin for every column
stationaryCols, min_d = tml.modeling.stationarity.min_ffd_all_cols(data)

# save to github for later 
min_dmin_d_save_for_backtesting = pd.Series(0, index=data.columns)
min_dmin_d_save_for_backtesting.update(min_d)
min_dmin_d_save_for_backtesting.dropna(inplace=True)
min_dmin_d_save_for_backtesting.to_csv(
    'C:/Users/Mislav/Documents/GitHub/trademl/data/min_d_' + contract + '.csv', sep=';')

# convert unstationary to stationary
data = tml.modeling.stationarity.unstat_cols_to_stat(data, min_d, stationaryCols)  # tml.modeling.stationarity.unstat_cols_to_stat
data.dropna(inplace=True)

# merge orig ohlc to spyStat
data = data.merge(ohlc, how='left', left_index=True, right_index=True)

  

### SAVE SPY WITH VIX

# save SPY
save_path = 'D:/market_data/usa/ohlcv_features/' + 'SPY' + '.h5'
with pd.HDFStore(save_path) as store:
    store.put('SPY', data)
