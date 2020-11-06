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
from sklearn.pipeline import make_pipeline
import mlfinlab as ml
from mlfinlab.feature_importance import get_orthogonal_features
import trademl as tml
import mfiles
from tensorboardX import SummaryWriter
from datetime import datetime
from pycaret.preprocess import Zroe_NearZero_Variance, Fix_multicollinearity
from trademl.modeling.structural_breaks import ChowStructuralBreakSubsample
from trademl.modeling.stationarity import StationarityMethod
from scipy.signal import savgol_filter
matplotlib.use("Agg")  # don't show graphs because thaty would stop guildai script
print('Start prepare step')


### TENSORBORADX WRITER
log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

### HYPERPARAMETERS
# load and save data
contract = 'SPY_IB'
input_data_path = 'D:/market_data/usa/ohlcv_features'
env_directory = None # os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
# subsample
chow_subsample = None
stationarity = 'orig'
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
tb_volatility_lookback = 50
tb_volatility_scaler = 1
time_step_length = 10
# train test split
train_test_split_ratio = 0.25
# feature engineering
choose_features = ['close']
correlation_threshold = 0.95
dim_reduction = 'none'
# scaling
scaling = 'expanding'
# performance
num_threads = 1


# Import data
file_name = contract + '_clean'
data = pd.read_hdf(os.path.join(Path(input_data_path), file_name + '.h5'), file_name)
data.sort_index(inplace=True)


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
    X = data.copy()
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
# if label_tuning
# else:
#     X_cols = [col for col in data.columns if 'day_' not in col]
#     X = data[X_cols]
#     y_cols = [col for col in data.columns if label + '_' in col]
#     labeling_info = data[y_cols]
#     # filtering
#     daily_vol = ml.util.get_daily_vol(data['orig_close' if 'orig_close' in data.columns else 'close'], lookback=50)
#     cusum_events = ml.filters.cusum_filter(data['orig_close' if 'orig_close' in data.columns else 'close'], threshold=daily_vol.mean()*1)
#     X = X.reindex(cusum_events)
#     labeling_info = labeling_info.reindex(cusum_events)

# remove na
remove_na_rows = labeling_info['bin'].isna()
X = X.loc[~remove_na_rows]
labeling_info = labeling_info.loc[~remove_na_rows]
labeling_info.loc[:, 'bin'] = np.where(
    labeling_info.loc[:, 'bin'] == -1, 
    0, 
    labeling_info.loc[:, 'bin'])

# Removing large values (TA issue - causes model problems / overflow)
X = X.loc[:, ~X.apply(lambda x: any((x >= 1e12) | (x <= -1e12)), axis=0)]

# Remove correlated assets
X = tml.modeling.preprocessing.remove_correlated_columns(
    data=X,
    columns_ignore=['close'],
    threshold=correlation_threshold)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labeling_info.loc[:, labeling_info.columns.str.contains('bin')],
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


### Add close if it does not exists, needed for later
if 'close' not in X_train.columns:
    X_train = X_train.join(X['close'], how='left')
    X_test = X_test.join(X['close'], how='left')


### SAVE FILES
# save localy   
file_names = ['X_train', 'y_train', 'X_test', 'y_test', 'labeling_info']
saved_files = [X_train, y_train, X_test, y_test, labeling_info]
file_names_pkl = [f + '.pkl' for f in file_names]
X_train.to_pickle('X_train.pkl')
tml.modeling.utils.save_files(
    saved_files,
    file_names_pkl,
    Path('./'))
X_train.to_pickle('X_train.pkl')
# save to mfiles
if env_directory is not None:
    mfiles_client = tml.modeling.utils.set_mfiles_client(env_directory)
    tml.modeling.utils.destroy_mfiles_object(mfiles_client, file_names)
    for f in file_names:
        mfiles_client.upload_file(file_names_pkl, object_type='Dokument')
print('End prepare step')
