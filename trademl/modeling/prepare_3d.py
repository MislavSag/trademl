from pathlib import Path
import os
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import mlfinlab as ml
from mlfinlab.feature_importance import get_orthogonal_features
import trademl as tml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import statsmodels.api as sm



### NON-MODEL HYPERPARAMETERS (for guildai)
# load and save data
input_data_path = 'D:/market_data/usa/ohlcv_features'
output_data_path = 'D:/algo_trading_files'
env_directory = None  # os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
# features
include_ta = True
# structural breaks
structural_break_regime = 'all'
# labeling
label_tuning = False
label = 'day_10'
labeling_technique = 'trend_scanning'
ts_look_forward_window = 240  # 60 * 8 * 10 (10 days)
ts_min_sample_length = 30
ts_step = 5
tb_triplebar_num_days = 10
tb_triplebar_pt_sl = [1, 1]
tb_triplebar_min_ret = 0.004
ts_look_forward_window = 1200  # 60 * 8 * 10 (10 days)
ts_min_sample_length = 30
ts_step = 5
tb_min_pct = 0.10
# filtering
tb_volatility_lookback = 100
tb_volatility_scaler = 1
# stationarity
stationarity_tecnique = 'fracdiff'
# feature engineering
correlation_threshold = 0.98
pca = False
# scaling
scaling = 'expanding'  # None
# performace
num_threads = 1
# sequence generation
train_val_index_split = 0.9



### IMPORT DATA
def import_data(data_path, remove_cols, contract='SPY'):
    # import data
    with pd.HDFStore(os.path.join(data_path, contract + '.h5')) as store:
        data = store.get(contract)
    data.sort_index(inplace=True)
    
    # remove variables
    remove_cols = [col for col in remove_cols if col in data.columns]
    data.drop(columns=remove_cols, inplace=True)
    
    return data


contract = 'SPY_raw_ta' if include_ta else 'SPY_raw'
contract = 'SPY_raw_ta' if label_tuning else contract + '_labels'
data = import_data(input_data_path, [], contract=contract)


### REGIME DEPENDENT ANALYSIS
if structural_break_regime == 'chow':
    if (data.loc[data['chow_segment'] == 1].shape[0] / 60 / 8) < 365:
        data = data.iloc[-(60*8*365):]
    else:
        data = data.loc[data['chow_segment'] == 1]
data = data.drop(columns=['chow_segment'])


### CHOOSE STATIONARY / UNSTATIONARY
if stationarity_tecnique == 'fracdiff':
    remove_cols = [col for col in data.columns if 'orig_' in col and col != 'orig_close']  
elif stationarity_tecnique == 'orig':
    remove_cols = [col for col in data.columns if 'fracdiff_' in col and col != 'orig_close']
data = data.drop(columns=remove_cols)


### CHOOSE SEQ LENGTH
close_column = ['fracdiff_close' if 'fracdiff_close' in data.columns else 'orig_close']
pacf = sm.tsa.stattools.pacf(data[close_column], nlags=200)
sig_test = lambda tau_h: np.abs(tau_h) > 2.58/np.sqrt(len(data))
for i in range(len(pacf)):
    if sig_test(pacf[i]) == False:
        time_step_length = i - 1
        print('time_step_length set to ', time_step_length)
        break


### LABELLING
if label_tuning:
    if labeling_technique == 'triple_barrier':
        # TRIPLE BARRIER LABELING
        triple_barrier_pipe= tml.modeling.pipelines.TripleBarierLabeling(
            close_name='orig_close' if 'orig_close' in data.columns else 'close',
            volatility_lookback=tb_volatility_lookback,
            volatility_scaler=tb_volatility_scaler,
            triplebar_num_days=tb_triplebar_num_days,
            triplebar_pt_sl=tb_triplebar_pt_sl,
            triplebar_min_ret=tb_triplebar_min_ret,
            num_threads=num_threads,
            tb_min_pct=tb_min_pct
        )   
        tb_fit = triple_barrier_pipe.fit(data)
        labeling_info = tb_fit.triple_barrier_info
        X = tb_fit.transform(data)
    elif labeling_technique == 'trend_scanning':
        trend_scanning_pipe = tml.modeling.pipelines.TrendScanning(
            close_name='orig_close' if 'orig_close' in data.columns else 'close',
            volatility_lookback=tb_volatility_lookback,
            volatility_scaler=tb_volatility_scaler,
            ts_look_forward_window=ts_look_forward_window,
            ts_min_sample_length=ts_min_sample_length,
            ts_step=ts_step
            )
        labeling_info = trend_scanning_pipe.fit(data)
        X = trend_scanning_pipe.transform(data)
    elif labeling_technique == 'fixed_horizon':
        X = data.copy()
        labeling_info = ml.labeling.fixed_time_horizon(data['orig_close'], threshold=0.005, resample_by='B').dropna().to_frame()
        labeling_info = labeling_info.rename(columns={'orig_close': 'bin'})
        print(labeling_info.iloc[:, 0].value_counts())
        X = X.iloc[:-1, :]
else:
    X_cols = [col for col in data.columns if 'day_' not in col]
    X = data[X_cols]
    y_cols = [col for col in data.columns if label in col]
    labeling_info = data[y_cols]


### FILTERING 
if not label_tuning:
    daily_vol = ml.util.get_daily_vol(data['orig_close' if 'orig_close' in data.columns else 'close'], lookback=50)
    cusum_events = ml.filters.cusum_filter(data['orig_close' if 'orig_close' in data.columns else 'close'], threshold=daily_vol.mean()*1)
else:
    daily_vol = ml.util.get_daily_vol(data['orig_close' if 'orig_close' in data.columns else 'close'], lookback=50)
    cusum_events = ml.filters.cusum_filter(data['orig_close' if 'orig_close' in data.columns else 'close'], threshold=daily_vol.mean()*1)


### REMOVE NA
remove_na_rows = labeling_info.isna().any(axis=1)
X = X.loc[~remove_na_rows]
labeling_info = labeling_info.loc[~remove_na_rows]
labeling_info.iloc[:, -1] = np.where(labeling_info.iloc[:, -1] == -1, 0, labeling_info.iloc[:, -1])
# labeling_info.iloc[:, -1] = labeling_info.iloc[:, -1].astype(pd.Int64Dtype())


### REMOVE CORRELATED ASSETS
msg = f'Shape before removing correlated features with threshold {correlation_threshold}' \
      f' is {X.shape} and after is'
print(msg)
X = tml.modeling.preprocessing.remove_correlated_columns(
    data=X,
    columns_ignore=[],
    threshold=correlation_threshold)
print(X.shape)


# TREAT CATEGORIAL VARIABLES
X = X.drop(columns=['tick_rule', 'HT_TRENDMODE', 'volume_vix'])  # remove for now



### TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, labeling_info.loc[:, labeling_info.columns.str.contains('bin')],
    test_size=0.15, shuffle=False, stratify=None)


### SCALING
if scaling == 'expanding':
    stdize_input = lambda x: (x - x.expanding(time_step_length).mean()) / x.expanding(time_step_length).std()
    X_train = X_train.apply(stdize_input)
    X_test = X_test.apply(stdize_input)
    y_train = y_train.loc[~X_train.isna().any(axis=1)]
    X_train = X_train.dropna()
    y_test = y_test.loc[~X_test.isna().any(axis=1)]
    X_test = X_test.dropna()
    


### 3D SEQUENCE
# save colnames fo later
col_names = X_train.columns
# calculate daily vol and filter time to trade
def sequence_from_array(data, target_vec, cusum_events, time_step_length):
    cusum_events_ = cusum_events.intersection(data.index)
    lstm_sequences = []
    targets = []
    for date in cusum_events_:
        observation = data[:date].iloc[-time_step_length:]
        if observation.shape[0] < time_step_length or data.index[-1] < date:
            next
        else:
            lstm_sequences.append(observation.values.reshape((1, observation.shape[0], observation.shape[1])))
            targets.append(target_vec[target_vec.index == date])
    lstm_sequences_all = np.vstack(lstm_sequences)
    targets = np.vstack(targets)
    targets = targets.astype(np.int64)
    return lstm_sequences_all, targets

# example
X_val, y_val = sequence_from_array(
    X_train.iloc[int((train_val_index_split*X_train.shape[0] + 1)):],
    y_train.iloc[int((train_val_index_split*X_train.shape[0] + 1)):],
    cusum_events, time_step_length)
X_train, y_train = sequence_from_array(
    X_train.iloc[:int(train_val_index_split*X_train.shape[0])],
    y_train.iloc[:int(train_val_index_split*X_train.shape[0])],
    cusum_events, time_step_length)
X_test, y_test = sequence_from_array(X_test, y_test, cusum_events, time_step_length)

# test for shapes
print('X and y shape train: ', X_train.shape, y_train.shape)
print('X and y shape validate: ', X_val.shape, y_val.shape)
print('X and y shape test: ', X_test.shape, y_test.shape)

# change labels type to integer64
y_train = y_train.astype(np.int64)
y_val = y_val.astype(np.int64)
y_test = y_test.astype(np.int64)


### SAVE FILES
# save localy
file_names = ['X_train_seq', 'y_train_seq', 'X_test_seq', 'y_test_seq', 'X_val_seq', 'y_val_seq']
saved_files = [X_train, y_train, X_test, y_test, X_val, y_val]
tml.modeling.utils.save_files(
    saved_files,
    file_names,
    output_data_path)
pd.Series(col_names).to_csv(os.path.join(Path(output_data_path), 'col_names.csv'))
# save to mfiles
if env_directory is not None:
    file_names = [f + '.npy' for f in file_names]
    mfiles_client = tml.modeling.utils.set_mfiles_client(env_directory)
    tml.modeling.utils.destroy_mfiles_object(mfiles_client, file_names)
    wd = os.getcwd()
    os.chdir(Path(output_data_path))
    for f in file_names:
        mfiles_client.upload_file(f, object_type='Dokument')
    os.chdir(wd)


### TEST IF SHAPES AND VALUES ARE RIGHT
# len(X_train[0,:,0]) == time_step_length
# first_series_manually = X[:cusum_events[2]]
# first_series_manually = first_series_manually[-time_step_length:]
# print((X_train[2,:,0] == first_series_manually.iloc[:, 0]).all())  # test for first series and first feature
# (X_train[0,:,1] == first_series_manually.iloc[:, 1]).all()  # test for first series and second feature
# second_series_manually = X[:cusum_events[1]]
# second_series_manually = second_series_manually[-25:]
# print((X_train[1,:,0] == second_series_manually.iloc[:, 0]).all())
# print((X_train[1,:,1] == second_series_manually.iloc[:, 1]).all())


### TEST MODEL
# X_train_test = X_train[:1000]
# y_train_test = y_train[:1000]
# X_val_test = X_val[:20]
# y_val_test = y_val[:20]
# model = keras.Sequential()
# model.add(layers.LSTM(32,
#                       return_sequences=True,
#                       input_shape=[None, X_train.shape[2]]))
# model.add(layers.LSTM(32, dropout=0.2))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',
#                 optimizer=keras.optimizers.Adam(),
#                 metrics=['accuracy',
#                         keras.metrics.AUC(),
#                         keras.metrics.Precision(),
#                         keras.metrics.Recall()]
#                 )
# history = model.fit(X_train_test, y_train_test, batch_size=128, epochs = 5, validation_data = (X_val_test, y_val_test))
