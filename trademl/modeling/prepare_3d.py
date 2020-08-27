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


### DON'T SHOW GRAPH OPTION (this is for guildai, ot to shoe graphs)
matplotlib.use("Agg")


### GLOBALS (path to partialy preprocessed data)
DATA_PATH = 'D:/market_data/usa/ohlcv_features/'

### NON-MODEL HYPERPARAMETERS (for guildai)
output_path = 'C:/Users/Mislav/Documents/GitHub/trademl/trademl/modeling/'
num_threads = 1
label = 'day_10'
structural_break_regime = 'all'
labeling_technique = 'trend_scanning'
std_outlier = 10
tb_volatility_lookback = 500
tb_volatility_scaler = 1
tb_triplebar_num_days = 10
tb_triplebar_pt_sl = [1, 1]
tb_triplebar_min_ret = 0.004
ts_look_forward_window = 240  # 60 * 8 * 10 (10 days)
ts_min_sample_length = 30
ts_step = 5
tb_min_pct = 0.10
sample_weights_type = 'returns'
stationary_close_lables = False
correlation_threshold = 0.98
pca = False


### IMPORT DATA
def import_data(data_path, remove_cols, contract='SPY'):
    # import data
    with pd.HDFStore(data_path + '/' + contract + '.h5') as store:
        data = store.get(contract)
    data.sort_index(inplace=True)
    
    # remove variables
    remove_cols = [col for col in remove_cols if col in data.columns]
    data.drop(columns=remove_cols, inplace=True)
    
    return data


remove_ohl = ['open', 'low', 'high', 'average', 'barCount',
              'open_vix', 'high_vix', 'low_vix', 'volume_vix']
data = import_data(DATA_PATH, remove_ohl, contract='SPY_raw')


### REGIME DEPENDENT ANALYSIS
if structural_break_regime == 'chow':
    if (data.loc[data['chow_segment'] == 1].shape[0] / 60 / 8) < 365:
        data = data.iloc[-(60*8*365):]
    else:
        data = data.loc[data['chow_segment'] == 1]
data = data.drop(columns=['chow_segment'])



### CHOOSE LABELLING TECHNIQUE
X_cols = [col for col in data.columns if 'day_' not in col]
X = data[X_cols]
y_cols = [col for col in data.columns if label in col]
y_matrix = data[y_cols]


### REMOVE NA
remove_na_rows = y_matrix.isna().any(axis=1)
X = X.loc[~remove_na_rows]
y_matrix = y_matrix.loc[~remove_na_rows]
y_matrix.iloc[:, -1] = np.where(y_matrix.iloc[:, -1] == -1, 0, y_matrix.iloc[:, -1])


### REMOVE CORRELATED ASSETS
X = tml.modeling.preprocessing.remove_correlated_columns(
    data=X,
    columns_ignore=[],
    threshold=correlation_threshold)


### TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y_matrix.loc[:, y_matrix.columns.str.contains('bin')],
    test_size=0.10, shuffle=False, stratify=None)


# PREPROCESSIONG
# scaling
stdize_input = lambda x: (x - x.expanding(time_step_length).mean()) / x.expanding(time_step_length).std()
X_train = X_train.apply(stdize_input)
X_test = X_test.apply(stdize_input)
y_train = y_train.loc[~X_train.isna().any(axis=1)]
X_train = X_train.dropna()
y_test = y_test.loc[~X_test.isna().any(axis=1)]
X_test = X_test.dropna()

# calculate daily vol and filter time to trade
daily_vol = ml.util.get_daily_vol(data['close'], lookback=50)
cusum_events = ml.filters.cusum_filter(data['close'], threshold=daily_vol.mean()*1)
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


X_train_seq, y_train_seq = sequence_from_array(
    X_train.iloc[:int(train_val_index_split*X_train.shape[0])],
    y_train.iloc[:int(train_val_index_split*X_train.shape[0])],
    cusum_events, time_step_length)
X_val_seq, y_val_seq = sequence_from_array(
    X_train.iloc[int((train_val_index_split*X_train.shape[0] + 1)):],
    y_train.iloc[int((train_val_index_split*X_train.shape[0] + 1)):],
    cusum_events, time_step_length)
X_test_seq, y_test_seq = sequence_from_array(X_test, y_test, cusum_events, time_step_length)


# test for shapes
print('X and y shape train: ', X_train_seq.shape, y_train_seq.shape)
print('X and y shape validate: ', X_val_seq.shape, y_val_seq.shape)
print('X and y shape test: ', X_test_seq.shape, y_test_seq.shape)



### TEST MODEL
model = keras.Sequential()
model.add(layers.LSTM(32,
                      return_sequences=True,
                      input_shape=[None, X_train.shape[1]]))
model.add(layers.LSTM(32, dropout=0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy',
                        keras.metrics.AUC(),
                        keras.metrics.Precision(),
                        keras.metrics.Recall()]
                )
history = model.fit(X_train_seq, y_train_seq, batch_size=batch_size, epochs = 20, validation_data = (X_val_seq, y_val_seq))


############# KERAS TEST ################
