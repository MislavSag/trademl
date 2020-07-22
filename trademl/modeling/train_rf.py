# fundamental modules
import numpy as np
import pandas as pd
from datetime import datetime
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
from pathlib import Path
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier
from sklearn.base import clone
import xgboost
import shap
import mlfinlab as ml
import trademl as tml
from tensorboardX import SummaryWriter
matplotlib.use("Agg")  # don't show graphs


### TENSORBORADX WRITER
log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S"
)
writer = SummaryWriter(log_dir)

### GLOBALS
DATA_PATH = 'D:/market_data/usa/ohlcv_features'


### NON-MODEL HYPERPARAMETERS
num_threads = 1
structural_break_regime = 'all'
labeling_technique = 'trend_scanning'
tb_volatility_lookback = 500
tb_volatility_scaler = 1
tb_triplebar_num_days = 10
tb_triplebar_pt_sl = [1, 1]
tb_triplebar_min_ret = 0.004
ts_look_forward_window = 1200  # 60 * 8 * 10 (10 days)
ts_min_sample_length = 30
ts_step = 5
tb_min_pct = 0.10
sample_weights_type = 'returns'
cv_type = 'purged_kfold'
cv_number = 4
rand_state = 3
stationary_close_lables = False

### PREPROCESSING PARAMETERS
correlation_threshold = 0.90

### MODEL HYPERPARAMETERS
max_depth = 3
max_features = 15
n_estimators = 500
min_weight_fraction_leaf = 0.05
class_weight = 'balanced_subsample'

### FEATURE SELECTION
keep_important_features = 20


def import_data(data_path, remove_cols, contract='SPY'):
    # import data
    with pd.HDFStore(data_path + '/' + contract + '.h5') as store:
        data = store.get(contract)
    data.sort_index(inplace=True)
    
    # remove variables
    remove_cols = [col for col in remove_cols if col in data.columns]
    data.drop(columns=remove_cols, inplace=True)
    
    return data


### IMPORT DATA
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
    data = remove_correlated_columns(data=data,
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


# # compare my and mlfinlab function
# daily_vol = ml.util.get_daily_vol(data['close_orig'], lookback=tb_volatility_lookback)
# cusum_events = ml.filters.cusum_filter(data['close_orig'], threshold=daily_vol.mean()*tb_volatility_scaler)
# test_test_lableing = ml.labeling.trend_scanning_labels(data['close_orig'], cusum_events,
#                                                 ts_look_forward_window, ts_min_sample_length, ts_step)

# my_test = trend_scanning_labels(data['close_orig'], cusum_events,
#                                                 ts_look_forward_window, ts_min_sample_length, ts_step)

# test_test_lableing.head(10)
# labeling_info.head(10)
# my_test.head(10)
# test_test_lableing.iloc[390:395]
# labeling_info.iloc[390:395]


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X.drop(columns=['close_orig']), labeling_info['bin'],
    test_size=0.10, shuffle=False, stratify=None)


### SAMPLE WEIGHTS (DECAY FACTOR CAN BE ADDED!)
if labeling_technique == 'trend_scanning':
    sample_weigths = labeling_info['t_value'].reindex(X_train.index).abs()
elif sample_weights_type == 'returns':
    sample_weigths = ml.sample_weights.get_weights_by_return(
        labeling_info.reindex(X_train.index),
        data.loc[X_train.index, 'close_orig'],
        num_threads=1)
elif sample_weights_type == 'time_decay':
    sample_weigths = ml.sample_weights.get_weights_by_time_decay(
        labeling_info.reindex(X_train.index),
        data.loc[X_train.index, 'close_orig'],
        decay=0.5, num_threads=1)
elif sample_weights_type == 'none':
    sample_weigths = None


### CROS VALIDATION STEPS
if cv_type == 'purged_kfold':
    cv = ml.cross_validation.PurgedKFold(
        n_splits=cv_number,
        samples_info_sets=labeling_info['t1'].reindex(X_train.index))


### MODEL
# MLDP str 98/99
clf = RandomForestClassifier(criterion='entropy',
                                max_features=max_features,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_depth=max_depth,
                                n_estimators=n_estimators,
                                class_weight=class_weight,
                                random_state=rand_state,
                                n_jobs=16)
scores = ml.cross_validation.ml_cross_val_score(
    clf, X_train, y_train, cv_gen=cv, 
    sample_weight_train=sample_weigths,
    scoring=sklearn.metrics.accuracy_score)  #sklearn.metrics.f1_score(average='weighted')

X_train.isna().any().any()
y_train.isna().any()
sample_weigths.isna().any()
np.isinf(y_train).any()
np.isinf(sample_weigths).any()
np.where(np.isinf(sample_weigths))

mean_score = scores.mean()
std_score = scores.std()
writer.add_scalar(tag='mean_score', scalar_value=mean_score, global_step=None)
writer.add_scalar(tag='std_score', scalar_value=std_score, global_step=None)
save_id = f'{max_depth}{max_features}{n_estimators}{str(mean_score)[2:6]}'

# retrain the model if mean score is high enough (higher than 0.5)
if mean_score < 0.55:
    print('good_performance: False')
else:
    print('good_performance: True')
    
    # refit the model and get results
    clf = RandomForestClassifier(criterion='entropy',
                                max_features=max_features,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_depth=max_depth,
                                n_estimators=n_estimators,
                                class_weight=class_weight,
                                random_state=rand_state,
                                n_jobs=16)
    clf.fit(X_train, y_train, sample_weight=sample_weigths)
    # tml.modeling.metrics_summary.clf_metrics(
    #     clf, X_train, X_test, y_train, y_test, avg='binary')
    # tml.modeling.metrics_summary.clf_metrics_tensorboard(
    #     clf, X_train, X_test, y_train, y_test, avg='binary')

    # save feature importance tables and plots
    shap_values, importances, mdi_feature_imp = tml.modeling.feature_importance.important_fatures(
        clf, X_train, y_train, plot_name=save_id)
    tml.modeling.utils.save_files([shap_values, importances, mdi_feature_imp],
                file_names=[f'shap_{save_id}.csv',
                            f'rf_importance_{save_id}.csv',
                            f'mpi_{save_id}.csv'],
                directory='important_features')
    
    
    # ### REFIT THE MODEL WITH MOST IMPORTANT FEATURES
    fi_cols = shap_values['col_name'].head(keep_important_features)
    X_train_important = X_train[fi_cols]
    X_test_important = X_test[fi_cols]
    clf_important = clf.fit(X_train_important, y_train)
    # tml.modeling.metrics_summary.clf_metrics(
    #     clf_important, X_train_important,
    #     X_test_important, y_train, y_test, avg='binary', prefix='fi_')
    # tml.modeling.metrics_summary.clf_metrics(
    #     clf_important, X_train_important,
    #     X_test_important, y_train, y_test, avg='binary', prefix='fi_')


# cose writer
writer.close()

# mean_ = data['close'].expanding(30).mean()
# std_ = data['close'].expanding(30).std()
# data['expanded_close'] = (data['close'] - mean_) / std_

