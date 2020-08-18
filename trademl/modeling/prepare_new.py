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

### MODEL HYPERPARAMETERS
input_path = 'C:/Users/Mislav/Documents/GitHub/trademl/trademl/modeling'
train_val_index_split = 0.75
time_step_length = 120
batch_size = 128
n_lstm_layers = 3
n_units = 64
dropout = 0.2
lr = 10e-2
epochs = 50
optimizer = 'random'
max_trials = 2  # parameter for random optimizer
executions_per_trial = 2  # parameter for random optimizer


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
              'open_vix', 'high_vix', 'low_vix']
data = import_data(DATA_PATH, remove_ohl, contract='SPY_raw')


### CHOOSE LABELLING TECHNIQUE
X_cols = [col for col in data.columns if 'day_' not in col]
X = data[X_cols]
y_cols = [col for col in data.columns if label in col]
y_matrix = data[y_cols]


### REMOVE NA
remove_na_rows = y_matrix.isna().any(axis=1)
X = X.loc[~remove_na_rows]
y_matrix = y_matrix.loc[~remove_na_rows]


### REMOVE CORRELATED ASSETS
def remove_correlated_columns(data, columns_ignore, threshold=0.99):
    # calculate correlation matrix
    corrs = pd.DataFrame(np.corrcoef(
        data.drop(columns=columns_ignore).values, rowvar=False),
                         columns=data.drop(columns=columns_ignore).columns)
    corrs.index = corrs.columns  # add row index
    # remove sequentally highly correlated features
    cols_remove = []
    for i, col in enumerate(corrs.columns):
        corrs_sample = corrs.iloc[i:, i:]  # remove ith column and row
        corrs_vec = corrs_sample[col].iloc[(i+1):]
        index_multicorr = corrs_vec.iloc[np.where(np.abs(corrs_vec) >= threshold)]
        cols_remove.append(index_multicorr)
    extreme_correlateed_assets = pd.DataFrame(cols_remove).columns
    data = data.drop(columns=extreme_correlateed_assets)

    return data


data = tml.modeling.preprocessing.remove_correlated_columns(
    data=X,
    columns_ignore=[],
    threshold=correlation_threshold)


### REGIME DEPENDENT ANALYSIS
if structural_break_regime == 'chow':
    if (data.loc[data['chow_segment'] == 1].shape[0] / 60 / 8) < 365:
        data = data.iloc[-(60*8*365):]
    else:
        data = data.loc[data['chow_segment'] == 1]


### TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y_matrix.loc[:, y_matrix.columns.str.contains('bin')],
    test_size=0.10, shuffle=False, stratify=None)


### PREPARE LSTM
x = X_train.values
y = y_train.values.reshape(-1, 1)
x_test = X_test.values
y_test_ = y_test.values.reshape(-1, 1)
train_generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data=x,
    targets=y,
    length=time_step_length,
    sampling_rate=1,
    stride=1,
    start_index=0,
    end_index=int(train_val_index_split*x.shape[0]),
    shuffle=False,
    reverse=False,
    batch_size=batch_size
)
validation_generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data=x,
    targets=y,
    length=time_step_length,
    sampling_rate=1,
    stride=1,
    start_index=int((train_val_index_split*x.shape[0] + 1)),
    end_index=None,  #int(train_test_index_split*X.shape[0])
    shuffle=False,
    reverse=False,
    batch_size=batch_size
)
test_generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data=x_test,
    targets=y_test_,
    length=time_step_length,
    sampling_rate=1,
    stride=1,
    start_index=0,
    end_index=None,
    shuffle=False,
    reverse=False,
    batch_size=batch_size
)


### FILTERING
daily_vol = ml.util.get_daily_vol(data['close'], lookback=50)
cusum_events = ml.filters.cusum_filter(data['close'], threshold=daily_vol.mean()*1)
train_filter = X_train.iloc[:int(train_val_index_split*x.shape[0])].index.isin(cusum_events)
val_filter = X_train.iloc[int((train_val_index_split*x.shape[0] + 1)):X_train.shape[0]].index.isin(cusum_events)
test_filter = X_test.index.isin(cusum_events)

# convert generator to inmemory 3D series (if enough RAM)
def generator_to_obj(generator, filter_vec):
    xlist = []
    ylist = []
    for i in range(len(generator)):
        if filter_vec[i]:
            x, y = generator[i]
            xlist.append(x)
            ylist.append(y)
    X_train = np.concatenate(xlist, axis=0)
    y_train = np.concatenate(ylist, axis=0)
    return X_train, y_train


def generator_to_obj(generator):
    xlist = []
    ylist = []
    for i in range(len(generator)):
        x, y = generator[i]
        xlist.append(x)
        ylist.append(y)
    X_train = np.concatenate(xlist, axis=0)
    y_train = np.concatenate(ylist, axis=0)
    return X_train, y_train


X_test_lstm, y_test_lstm = generator_to_obj(test_generator)
X_val_lstm, y_val_lstm = generator_to_obj(validation_generator)

X_train_lstm, y_train_lstm = generator_to_obj(train_generator)
X_val_lstm, y_val_lstm = generator_to_obj(validation_generator)
X_test_lstm, y_test_lstm = generator_to_obj(test_generator, test_filter)

# test for shapes
print('X and y shape train: ', X_train_lstm.shape, y_train_lstm.shape)
print('X and y shape validate: ', X_val_lstm.shape, y_val_lstm.shape)
print('X and y shape test: ', X_test_lstm.shape, y_test_lstm.shape)

# change -1 to 1
for i, y in enumerate(y_train_lstm):
    if y == -1.:
        y_train_lstm[i,:] = 0. 
for i, y in enumerate(y_val_lstm):
    if y == -1.:
        y_val_lstm[i,:] = 0. 
for i, y in enumerate(y_test_lstm):
    if y == -1.:
        y_test_lstm[i,:] = 0. 

# change labels type to integer64
y_train_lstm = y_train_lstm.astype(np.int64)
y_val_lstm = y_val_lstm.astype(np.int64)
y_test_lstm = y_test_lstm.astype(np.int64)