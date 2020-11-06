from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import mlfinlab as ml
from mlfinlab.feature_importance import get_orthogonal_features
import trademl as tml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import statsmodels.api as sm
from pycaret.preprocess import Zroe_NearZero_Variance, Fix_multicollinearity
from trademl.modeling.structural_breaks import ChowStructuralBreakSubsample
from trademl.modeling.stationarity import StationarityMethod
from tensorboardX import SummaryWriter
from datetime import datetime



# Tensorbaord writer
log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)


# Hyperparamteres
# load and save data
contract = 'SPY_IB'
input_data_path = 'D:/market_data/usa/ohlcv_features'
env_directory = None # os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
# subsample
chow_subsample = None
stationarity = 'orig'
# filtering
filtering = None  # cusum
# labeling
labeling_technique = 'tl'  # tb is triple-barrier; ts is trend scanning; tl is trend labaling
tb_triplebar_num_days = 2
tb_triplebar_pt_sl = [1, 1]
tb_triplebar_min_ret = 0.004
ts_look_forward_window = 1200  # 60 * 8 * 10 (10 days)
ts_min_sample_length = 30
ts_step = 5
tb_min_pct = 0.05
w = 0.15
# filtering
tb_volatility_lookback = 10
tb_volatility_scaler = 1
time_step_length = 10
# train test split
train_test_split_ratio = 0.25
# feature engineering
choose_features = ['close']
correlation_threshold = 0.95
dim_reduction = 'none'
# scaling
scaling = 'expanding_mean'
# performance
num_threads = 1
# sequence generation
train_val_index_split = 0.9


# Import data
file_name = contract + '_clean'
data = pd.read_hdf(os.path.join(Path(input_data_path), file_name + '.h5'), file_name)
data.sort_index(inplace=True)


# Choose columns
if 'close' not in choose_features:
    sys.exit("Data must have close column")
data = data[choose_features]


# Remove constant columns
data = data.loc[:, data.apply(pd.Series.nunique) != 1]


# Choose sequence length
pacf = sm.tsa.stattools.pacf(data['close'], nlags=200)
sig_test = lambda tau_h: np.abs(tau_h) > 2.58/np.sqrt(len(data))
for i in range(len(pacf)):
    if sig_test(pacf[i]) == False:
        time_step_length = i - 1
        print('time_step_length set to ', time_step_length)
        break
time_step_length = 5 if time_step_length < 5 else time_step_length

# Choose subsamples, stationarity method and make labels
pipe = make_pipeline(
    ChowStructuralBreakSubsample(min_length=10) if chow_subsample else None,
    StationarityMethod(stationarity_method=stationarity),
    )
data = pipe.fit_transform(data)


# categorical variables
categorial_features = ['tick_rule', 'HT_TRENDMODE', 'volume_vix']
categorial_features = [col for col in categorial_features if col in data.columns]
data = data.drop(columns=categorial_features)  # remove for now


# filtering
if filtering == 'cusum':
    daily_vol = ml.util.get_daily_vol(
        data['close'], lookback=50)
    cusum_events = ml.filters.cusum_filter(
        data['close'], threshold=daily_vol.mean()*1)
else:
    cusum_events = data.index

# Labeling_
if labeling_technique == 'tl':
    labeling_info = tml.modeling.labeling.trend_labeling(
        close=data['close'].to_list(),
        time=data.index.to_list(),
        w=w)
    labeling_info = pd.DataFrame(labeling_info, index=data.index, columns=['bin'])
    labeling_info['t1'] = np.nan
    labeling_info['ret'] = np.nan
    labeling_info['trgt'] = np.nan
if labeling_technique == 'tb':
    triple_barrier_pipe= tml.modeling.pipelines.TripleBarierLabeling(
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
elif labeling_technique == 'ts':
    trend_scanning_pipe = tml.modeling.pipelines.TrendScanning(
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


# choose X and Y
X = data.copy()
Y = labeling_info.copy()

# Remove NA
remove_na_rows = Y['bin'].isna()
X = X.loc[~remove_na_rows, :]
Y = Y.loc[~remove_na_rows]
Y.loc[:, 'bin'] = np.where(Y.loc[:, 'bin'] == -1, 0, Y.loc[:, 'bin'])


# Removing large values (TA issue - causes model problems / overflow)
X = X.loc[:, ~X.apply(lambda x: any((x >= 1e12) | (x <= -1e12)), axis=0)]


# Remove correlated assets
msg = f'Shape before removing correlated features with threshold {correlation_threshold}' \
      f' is {X.shape} and after is'
print(msg)
X = tml.modeling.preprocessing.remove_correlated_columns(
    data=X,
    columns_ignore=['close'],
    threshold=correlation_threshold)
print(X.shape)


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y.loc[:, Y.columns.str.contains('bin')],
    test_size=train_test_split_ratio, shuffle=False, stratify=None)


##### GENETICS INDICATORS
# x = y_train.squeeze()
# x.value_counts()
# add genetics indicators
# if dim_reduction == 'gplearn':
#     gen = tml.modeling.features.Genetic()
#     gen.fit(X_train, y_train)
    
#     test = X.predict(X_test)
##### GENETICS INDICATORS


# Scaling
def scale_expanding(X_train, y_train, X_test, y_test, expand_function):
    X_train = X_train.apply(expand_function)
    X_test = X_test.apply(expand_function)
    y_train = y_train.loc[~X_train.isna().any(axis=1)]
    X_train = X_train.dropna()
    y_test = y_test.loc[~X_test.isna().any(axis=1)]
    X_test = X_test.dropna()

    return X_train, y_train, X_test, y_test


if scaling == 'expanding':
    f = lambda x: (x - x.rolling(tb_volatility_lookback).mean()) / x.rolling(tb_volatility_lookback).std()
    X_train, y_train, X_test, y_test = scale_expanding(
        X_train, y_train, X_test, y_test, f)    
elif scaling == 'expanding_mean':
    f = lambda x: (x - x.rolling(tb_volatility_lookback).mean())
    X_train, y_train, X_test, y_test = scale_expanding(
        X_train, y_train, X_test, y_test, f)    


# Dimensionality reduction
if dim_reduction == 'pca':
    if scaling == 'none':
        X_train = pd.DataFrame(preprocessing.scale(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(preprocessing.scale(X_test), columns=X_test.columns)
    X_train = pd.DataFrame(
        get_orthogonal_features(X_train),
        index=X_train.index).add_prefix("PCA_")
    pca_n_compenents = X_train.shape[1]
    X_test = pd.DataFrame(
        get_orthogonal_features(X_test,
                                num_features=pca_n_compenents),
        index=X_test.index).add_prefix("PCA_")
    X_train.index = y_train.index
    X_test.index = y_test.index
# elif dim_reduction == 'gplearn':
#     gen = Genetic()
#     X = gen.fit_transform(X, labeling_info.loc[:, labeling_info.columns.str.contains('bin')])


# Add close if it does not exists, needed for later
if 'close' not in X_train.columns:
    X_train = X_train.join(X['close'], how='left')
    X_test = X_test.join(X['close'], how='left')


# Save column names and Y
pd.Series(X_train.columns).to_csv('col_names.csv')
Y.to_pickle('Y.pkl')


# Make 3D sequences from matrix
def sequence_from_array(data, target_vec, cusum_events, time_step_length):
    cusum_events_ = cusum_events.intersection(data.index)
    lstm_sequences = []
    targets = []
    for date in cusum_events_:
        observation = data[:date].iloc[-time_step_length:]
        if observation.shape[0] < time_step_length or data.index[-1] < date:
            next
        else:
            d3 = observation.values.reshape((1, observation.shape[0], observation.shape[1]))
            lstm_sequences.append(d3)
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


# Save files
# Save localy
file_names = ['X_train', 'y_train', 'X_test', 'y_test', 'X_val', 'y_val']
saved_files = [X_train, y_train, X_test, y_test, X_val, y_val]
[np.save(name, obj) for name, obj in zip(file_names, saved_files)]
# Save to mfiles
# if env_directory is not None:
#     file_names = [f + '.npy' for f in file_names]
#     mfiles_client = tml.modeling.utils.set_mfiles_client(env_directory)
#     tml.modeling.utils.destroy_mfiles_object(mfiles_client, file_names)
#     wd = os.getcwd()
#     os.chdir(Path(output_data_path))
#     for f in file_names:
#         mfiles_client.upload_file(f, object_type='Dokument')
#     os.chdir(wd)
print('End prepare step')


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
