# fundamental modules
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
import joblib
import json
import sys
# preprocessing
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import shap
# metrics 
import mlfinlab as ml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    log_loss,
    )
from boruta import BorutaPy
# finance packagesb
import mlfinlab as ml
import trademl as tml
import vectorbt as vbt



### DON'T SHOW GRAPH OPTION
matplotlib.use("Agg")


### GLOBALS
DATA_PATH = 'D:/market_data/usa/ohlcv_features/'
# NUMEXPR_MAX_THREADS = 10

### IMPORT DATA
contract = ['SPY']
with pd.HDFStore(DATA_PATH + contract[0] + '.h5') as store:
    data = store.get(contract[0])
data.sort_index(inplace=True)


### CHOOSE/REMOVE VARIABLES
remove_ohl = ['open', 'low', 'high', 'average', 'barCount',
              'vixFirst', 'vixHigh', 'vixLow', 'vixClose',
              'vixVolume', 'open_orig', 'high_orig', 'low_orig']
remove_ohl = [col for col in remove_ohl if col in data.columns]
data.drop(columns=remove_ohl, inplace=True)  #correlated with close


### NON-MODEL HYPERPARAMETERS
labeling_technique = 'triple_barrier'
std_outlier = 10
tb_volatility_lookback = 50
tb_volatility_scaler = 1
tb_triplebar_num_days = 30
tb_triplebar_pt_sl = [1, 1]
tb_triplebar_min_ret = 0.005
ts_look_forward_window = 4800  # 60 * 8 * 10 (10 days) = 4800
ts_min_sample_length = 30
ts_step = 5
tb_min_pct = 0.10
sample_weights_type = 'returns'
cv_type = 'purged_kfold'
cv_number = 4
rand_state = 3
remove_ind_with_high_period = True
stationary_close_lables = False

### MODEL HYPERPARAMETERS
max_depth=3
max_features = 15
n_estimators = 1000

### POSTMODEL PARAMETERS
keep_important_features = 25
vectorbt_slippage = 0.0015
vectorbt_fees = 0.0015

### USE STATIONARY CLOSE TO CALCULATE LABELS
if stationary_close_lables:
    data['close_orig'] = data['close']  # with original close reslts are pretty bad!


### REMOVE INDICATORS WITH HIGH PERIOD
if remove_ind_with_high_period:
    data.drop(columns=['DEMA96000', 'ADX96000', 'TEMA96000',
                       'ADXR96000', 'TRIX96000'], inplace=True)
    data.drop(columns=['autocorr_1', 'autocorr_2', 'autocorr_3',
                       'autocorr_4', 'autocorr_5'], inplace=True)
    print('pass')
    

### REMOVE OUTLIERS
outlier_remove = tml.modeling.pipelines.OutlierStdRemove(std_outlier)
data = outlier_remove.fit_transform(data)


### LABELING
if labeling_technique == 'triple_barrier':
    # TRIPLE BARRIER LABELING
    triple_barrier_pipe= tml.modeling.pipelines.TripleBarierLabeling(
        close_name='close_orig',
        volatility_lookback=tb_volatility_lookback,
        volatility_scaler=tb_volatility_scaler,
        triplebar_num_days=tb_triplebar_num_days,
        triplebar_pt_sl=tb_triplebar_pt_sl,
        triplebar_min_ret=tb_triplebar_min_ret,
        num_threads=1,
        tb_min_pct=tb_min_pct
    )
    tb_fit = triple_barrier_pipe.fit(data)
    labeling_info = tb_fit.triple_barrier_info
    X = tb_fit.transform(data)
elif labeling_technique == 'trend_scanning':
    trend_scanning_pipe = tml.modeling.pipelines.TrendScanning(
        close_name='close_orig',
        volatility_lookback=tb_volatility_lookback,
        volatility_scaler=tb_volatility_scaler,
        ts_look_forward_window=ts_look_forward_window,
        ts_min_sample_length=ts_min_sample_length,
        ts_step=ts_step
        )
    labeling_info = trend_scanning_pipe.fit(data)
    X = trend_scanning_pipe.transform(data)

###### TEST
X_TEST = X.iloc[:2000]
labeling_info_TEST = labeling_info.iloc[:2000]
###### TEST

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_TEST.drop(columns=['close_orig']), labeling_info_TEST['bin'],
    test_size=0.10, shuffle=False, stratify=None)
    

### SAMPLE WEIGHTS (DECAY FACTOR CAN BE ADDED!)
if sample_weights_type == 'returns':
    sample_weigths = ml.sample_weights.get_weights_by_return(
        labeling_info_TEST.reindex(X_train.index),
        data.loc[X_train.index, 'close_orig'],
        num_threads=1)
elif sample_weights_type == 'time_decay':
    sample_weigths = ml.sample_weights.get_weights_by_time_decay(
        labeling_info.reindex(X_train.index),
        data.loc[X_train.index, 'close_orig'],
        decay=0.5, num_threads=1)
elif labeling_technique is 'trend_scanning':
    sample_weigths = labeling_info['t_value'].reindex(X_train.index).abs()


### PREPARE LSTM
# X_train_lstm = .drop(columns=['close_orig']).values
x = X_train['close'].values.reshape(-1, 1)
y = y_train.values.reshape(-1, 1)
# y = y.astype(str)
x_test = X_test['close'].values.reshape(-1, 1)
y_test_ = y_test.values.reshape(-1, 1)

train_val_index_split = 0.75
train_generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data=x,
    targets=y,
    length=30,
    sampling_rate=1,
    stride=1,
    start_index=0,
    end_index=int(train_val_index_split*X_TEST.shape[0]),
    shuffle=False,
    reverse=False,
    batch_size=128
)
validation_generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data=x,
    targets=y,
    length=30,
    sampling_rate=1,
    stride=1,
    start_index=int((train_val_index_split*X_TEST.shape[0] + 1)),
    end_index=None,  #int(train_test_index_split*X.shape[0])
    shuffle=False,
    reverse=False,
    batch_size=128
)
test_generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data=x_test,
    targets=y_test,
    length=30,
    sampling_rate=1,
    stride=1,
    start_index=0,
    end_index=None,
    shuffle=False,
    reverse=False,
    batch_size=128
)

# convert generator to inmemory 3D series (if enough RAM)
def generator_to_obj(generator):
    xlist = []
    ylist = []
    for i in range(len(generator)):
        x, y = train_generator[i]
        xlist.append(x)
        ylist.append(y)
    X_train = np.concatenate(xlist, axis=0)
    y_train = np.concatenate(ylist, axis=0)
    return X_train, y_train
    
X_train_lstm, y_train_lstm = generator_to_obj(train_generator)
X_val_lstm, y_val_lstm = generator_to_obj(validation_generator)
X_test_lstm, y_test_lstm = generator_to_obj(test_generator)

# test for shapes
print('X and y shape train: ', X_train_lstm.shape, y_train_lstm.shape)
print('X and y shape validate: ', X_val_lstm.shape, y_val_lstm.shape)
print('X and y shape test: ', X_test_lstm.shape, y_test_lstm.shape)


### MODEL
model = keras.models.Sequential([
        keras.layers.LSTM(124, return_sequences=True, input_shape=[None, 1]),
        keras.layers.LSTM(258),
        keras.layers.Dense(1, activation='sigmoid')
        
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=128,
                    validation_data=[X_val_lstm, y_val_lstm])
# history = model.fit_generator(train_generator, epochs=40, validation_data=validation_generator, verbose=1)
score, acc = model.evaluate(X_val_lstm, y_val_lstm,
                            batch_size=128)

historydf = pd.DataFrame(history.history)
historydf.head(10)



####### STACKOVERFLOW
X_TEST.to_csv('X_TEST.csv', sep=',')
labeling_info_TEST.to_csv('labeling_info_TEST.csv', sep=',')

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np

url = 'https://raw.githubusercontent.com/MislavSag/trademl/master/trademl/modeling/random_forest/X_TEST.csv'
X_TEST = pd.read_csv(url, sep=',')
url = 'https://raw.githubusercontent.com/MislavSag/trademl/master/trademl/modeling/random_forest/labeling_info_TEST.csv'
labeling_info_TEST = pd.read_csv(url, sep=',')


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_TEST.drop(columns=['close_orig']), labeling_info_TEST['bin'],
    test_size=0.10, shuffle=False, stratify=None)
    

### PREPARE LSTM
x = X_train['close'].values.reshape(-1, 1)
y = y_train.values.reshape(-1, 1)
x_test = X_test['close'].values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
train_val_index_split = 0.75
train_generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data=x,
    targets=y,
    length=30,
    sampling_rate=1,
    stride=1,
    start_index=0,
    end_index=int(train_val_index_split*X_TEST.shape[0]),
    shuffle=False,
    reverse=False,
    batch_size=128
)
validation_generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data=x,
    targets=y,
    length=30,
    sampling_rate=1,
    stride=1,
    start_index=int((train_val_index_split*X_TEST.shape[0] + 1)),
    end_index=None,  #int(train_test_index_split*X.shape[0])
    shuffle=False,
    reverse=False,
    batch_size=128
)
test_generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data=x_test,
    targets=y_test,
    length=30,
    sampling_rate=1,
    stride=1,
    start_index=0,
    end_index=None,
    shuffle=False,
    reverse=False,
    batch_size=128
)

# convert generator to inmemory 3D series (if enough RAM)
def generator_to_obj(generator):
    xlist = []
    ylist = []
    for i in range(len(generator)):
        x, y = train_generator[i]
        xlist.append(x)
        ylist.append(y)
    X_train = np.concatenate(xlist, axis=0)
    y_train = np.concatenate(ylist, axis=0)
    return X_train, y_train
    
X_train_lstm, y_train_lstm = generator_to_obj(train_generator)
X_val_lstm, y_val_lstm = generator_to_obj(validation_generator)
X_test_lstm, y_test_lstm = generator_to_obj(test_generator)

# test for shapes
print('X and y shape train: ', X_train_lstm.shape, y_train_lstm.shape)
print('X and y shape validate: ', X_val_lstm.shape, y_val_lstm.shape)
print('X and y shape test: ', X_test_lstm.shape, y_test_lstm.shape)

####### STACKOVERFLOW