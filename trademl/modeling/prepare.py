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


### TENSORFLOW ATTRIBUTES
assert tf.config.list_physical_devices('GPU')
assert tf.config.list_physical_devices('GPU')[0][1] == 'GPU'
assert tf.test.is_built_with_cuda()


### DON'T SHOW GRAPH OPTION (this is for guildai, ot to shoe graphs)
matplotlib.use("Agg")


### GLOBALS (path to partialy preprocessed data)
DATA_PATH = 'D:/market_data/usa/ohlcv_features/'


### NON-MODEL HYPERPARAMETERS (for guildai)
num_threads = 1
structural_break_regime = 'all'
labeling_technique = 'trend_scanning'
std_outlier = 10
tb_volatility_lookback = 500
tb_volatility_scaler = 1
tb_triplebar_num_days = 10
tb_triplebar_pt_sl = [1, 1]
tb_triplebar_min_ret = 0.004
ts_look_forward_window = 480  # 60 * 8 * 10 (10 days)
ts_min_sample_length = 30
ts_step = 5
tb_min_pct = 0.10
sample_weights_type = 'returns'
stationary_close_lables = False
correlation_threshold = 0.99
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
                'open_vix', 'high_vix', 'low_vix', 'close_vix', 'volume_vix',
                'open_orig', 'high_orig', 'low_orig']
data = import_data(DATA_PATH, remove_ohl, contract='SPY')


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


### REMOVE CORRELATED FEARURES
if correlation_threshold < 0.99:
    data = tml.modeling.preprocessing.remove_correlated_columns(
        data=data,
        columns_ignore=['close_orig'],
        threshold=correlation_threshold)


### REGIME DEPENDENT ANALYSIS
if structural_break_regime == 'chow':
    if (data.loc[data['chow_segment'] == 1].shape[0] / 60 / 8) < 365:
        data = data.iloc[-(60*8*365):]
    else:
        data = data.loc[data['chow_segment'] == 1]


### USE STATIONARY CLOSE TO CALCULATE LABELS
if stationary_close_lables:
    data['close_orig'] = data['close']  # with original close reslts are pretty bad!


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
        num_threads=num_threads,
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
elif labeling_technique == 'fixed_horizon':
    X = data.copy()
    labeling_info = ml.labeling.fixed_time_horizon(data['close_orig'], threshold=0.005, resample_by='B').dropna().to_frame()
    labeling_info = labeling_info.rename(columns={'close_orig': 'bin'})
    print(labeling_info.iloc[:, 0].value_counts())
    X = X.iloc[:-1, :]


### TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X.drop(columns=['close_orig']), labeling_info['bin'],
    test_size=0.10, shuffle=False, stratify=None)


### SAMPLE WEIGHTS
if labeling_technique == 'trend_scanning':
    sample_weights = labeling_info['t_value'].reindex(X_train.index).abs()
elif sample_weights_type == 'returns':
    sample_weights = ml.sample_weights.get_weights_by_return(
        labeling_info.reindex(X_train.index),
        data.loc[X_train.index, 'close_orig'],
        num_threads=1)
elif sample_weights_type == 'time_decay':
    sample_weights = ml.sample_weights.get_weights_by_time_decay(
        labeling_info.reindex(X_train.index),
        data.loc[X_train.index, 'close_orig'],
        decay=0.5, num_threads=1)
elif sample_weights_type == 'none':
    sample_weights = None


### DIMENSIONALITY REDUCTION
if pca:   
    X_train = pd.DataFrame(preprocessing.scale(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(preprocessing.scale(X_test), columns=X_test.columns)
    X_train = pd.DataFrame(
        get_orthogonal_features(
            X_train.drop(columns=['tick_rule', 'HT_TRENDMODE', 'chow_segment'])),
        index=X_train.index).add_prefix("PCA_")
    pca_n_compenents = X_train.shape[1]
    X_test = pd.DataFrame(
        get_orthogonal_features(
            X_test.drop(columns=['tick_rule', 'HT_TRENDMODE', 'chow_segment']),
            num_features=pca_n_compenents),
        index=X_test.index).add_prefix("PCA_")


### SAVE DATAFRAMES
def save_files(objects, file_names, directory='important_features'):            
    """
    Save file to specific deirectory.    
    params
    """
    # create directory if it does not exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # save files to directory
    for df, file_name in zip(objects, file_names):
        saving_path = Path(f'{directory}/{file_name}')
        if ".csv" in file_name: 
            df.to_csv(saving_path)
        elif ".pkl" in file_name:
            df.to_pickle(saving_path)


save_files([X_train, y_train, X_test, y_test, sample_weights, labeling_info],
           ['X_train.pkl', 'y_train.pkl', 'X_test.pkl',
            'y_test.pkl', 'sample_weights.pkl', 'labeling_info.pkl'],
           'data_prepare')
