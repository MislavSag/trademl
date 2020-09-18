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
import mfiles
from tensorboardX import SummaryWriter
from datetime import datetime
matplotlib.use("Agg")  # don't show graphs because thaty would stop guildai script


### TENSORBORADX WRITER
log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)


# import numpy as np
# import requests
# import json

# x = {
#     'x': np.random.rand(500).tolist(),
#     'adf_lag': 2
# }
# x = json.dumps(x)

# res = requests.post("http://46.101.219.193/plumber_test/radf", data=x)

# # res_json = res.json()
# # bsadf = res_json['bsadf']
# # bsadf = pd.DataFrame.from_dict(bsadf)
# # bsadf_last = bsadf.iloc[-1]



### HYPERPARAMETERS
# load and save data
input_data_path = 'D:/market_data/usa/ohlcv_features'
output_data_path = 'D:/algo_trading_files'
env_directory = None # os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
# features
include_ta = True
# stationarity
stationarity_tecnique = 'fracdiff'
# structural breaks
structural_break_regime = 'all'
# labeling
label_tuning = False
label = 'day_30'  # 'day_1' 'day_2' 'day_5' 'day_10' 'day_20' 'day_30' 'day_60'
labeling_technique = 'trend_scanning'
tb_triplebar_num_days = 10
tb_triplebar_pt_sl = [1, 1]
tb_triplebar_min_ret = 0.004
ts_look_forward_window = 1200  # 60 * 8 * 10 (10 days)
ts_min_sample_length = 30
ts_step = 5
tb_min_pct = 0.10
# filtering
tb_volatility_lookback = 50
tb_volatility_scaler = 1
# feature engineering
correlation_threshold = 0.99
pca = True
# scaling
scaling = 'none'
# performance
num_threads = 1



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
    y_cols = [col for col in data.columns if label + '_' in col]
    labeling_info = data[y_cols]


### FILTERING
if label_tuning:
    X = X.drop(columns=['orig_close'])
else:
    daily_vol = ml.util.get_daily_vol(data['orig_close' if 'orig_close' in data.columns else 'close'], lookback=50)
    cusum_events = ml.filters.cusum_filter(data['orig_close' if 'orig_close' in data.columns else 'close'], threshold=daily_vol.mean()*1)
    X = X.reindex(cusum_events)
    labeling_info = labeling_info.reindex(cusum_events)
### ZAVRSITI DO KRAJA ####


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
categorial_features = ['tick_rule', 'HT_TRENDMODE', 'volume_vix']
categorial_features = [col for col in categorial_features if col in X.columns]
X = X.drop(columns=categorial_features)  # remove for now


### TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, labeling_info.loc[:, labeling_info.columns.str.contains('bin')],
    test_size=0.10, shuffle=False, stratify=None)


### SCALING
if scaling == 'expanding':
    stdize_input = lambda x: (x - x.expanding(tb_volatility_lookback).mean()) / x.expanding(tb_volatility_lookback).std()
    X_train = X_train.apply(stdize_input)
    X_test = X_test.apply(stdize_input)
    y_train = y_train.loc[~X_train.isna().any(axis=1)]
    X_train = X_train.dropna()
    y_test = y_test.loc[~X_test.isna().any(axis=1)]
    X_test = X_test.dropna()

    y_train = y_train.loc[~X_train.isna().any(axis=1)]
    X_train = X_train.dropna()
    y_test = y_test.loc[~X_test.isna().any(axis=1)]
    X_test = X_test.dropna()

    
### DIMENSIONALITY REDUCTION
if pca:
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


### SAVE FILES
# save localy
file_names = ['X_train', 'y_train', 'X_test',
                'y_test', 'labeling_info']
saved_files = [X_train, y_train, X_test, y_test, labeling_info]
# if pca:
#     file_names = [f + '_pca' for f in file_names]
file_names_pkl = [f + '.pkl' for f in file_names]
[os.path.exists(os.path.join(output_data_path, f)) for f in file_names_pkl if os.path.exists(os.path.join(output_data_path, f))]
tml.modeling.utils.save_files(
    saved_files,
    file_names_pkl,
    output_data_path)
# file_names_csv = [f + '.csv' for f in file_names]
# tml.modeling.utils.save_files(
#     saved_files,
#     file_names_csv,
#     output_data_path)
# save to mfiles
if env_directory is not None:
    file_names = file_names_pkl #  + file_names_csv
    mfiles_client = tml.modeling.utils.set_mfiles_client(env_directory)
    tml.modeling.utils.destroy_mfiles_object(mfiles_client, file_names)
    wd = os.getcwd()
    os.chdir(Path(output_data_path))
    for f in file_names:
        mfiles_client.upload_file(f, object_type='Dokument')
    os.chdir(wd)
