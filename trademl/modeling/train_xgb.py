# fundamental modules
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
import joblib
import json
import sys
import os
# preprocessing
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import clone
import xgboost as xgb
import shap
# metrics 
import mlfinlab as ml
# finance packages
import trademl as tml


### DON'T SHOW GRAPH OPTION
matplotlib.use("Agg")


### GLOBALS
DATA_PATH = 'D:/market_data/usa/ohlcv_features/'


### IMPORT DATA
contract = ['SPY']
with pd.HDFStore(DATA_PATH + contract[0] + '.h5') as store:
    data = store.get(contract[0])
data.sort_index(inplace=True)


### CHOOSE/REMOVE VARIABLES
remove_ohl = ['open', 'low', 'high', 'average', 'barCount',
            # 'vixFirst', 'vixHigh', 'vixLow', 'vixClose', 'vixVolume',
            'open_orig', 'high_orig', 'low_orig']
remove_ohl = [col for col in remove_ohl if col in data.columns]
data.drop(columns=remove_ohl, inplace=True)  #correlated with close


### NON-MODEL HYPERPARAMETERS
num_threads = 1
structural_break_regime = 'all'
labeling_technique = 'trend_scanning'
std_outlier = 10
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

### MODEL HYPERPARAMETERS
# max_depth = 3
# max_features = 20
# n_estimators = 500

### POSTMODEL PARAMETERS
keep_important_features = 25
# vectorbt_slippage = 0.0015
# vectorbt_fees = 0.0015


### REGIME DEPENDENT ANALYSIS
if structural_break_regime == 'chow':
    if (data.loc[data['chow_segment'] == 1].shape[0] / 60 / 8) < 365:
        data = data.iloc[-(60*8*365):]
    else:
        data = data.loc[data['chow_segment'] == 1]

### USE STATIONARY CLOSE TO CALCULATE LABELS
if stationary_close_lables:
    data['close_orig'] = data['close']  # with original close reslts are pretty bad!


### REMOVE OUTLIERS
# outlier_remove = tml.modeling.pipelines.OutlierStdRemove(std_outlier)
# data = outlier_remove.fit_transform(data)


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


### CLUSTERED FEATURES
# feat_subs = ml.clustering.feature_clusters.get_feature_clusters(
#     X, dependence_metric='information_variation',
#     distance_metric='angular', linkage_method='singular',
#     n_clusters=1)


### CALENDARS
# import pandas_market_calendars as mcal
# # Create a calendar
# nyse = mcal.get_calendar('NYSE')
# schedule = nyse.schedule(start_date='2016-12-30', end_date='2017-01-10')
# schedule  
# # Show available calendars
# print(mcal.get_calendar_names())


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X.drop(columns=['close_orig']), labeling_info['bin'],
    test_size=0.10, shuffle=False, stratify=None)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, shuffle=False, stratify=None)


### SAMPLE WEIGHTS (DECAY FACTOR CAN BE ADDED!)
if sample_weights_type == 'returns':
    sample_weigths = ml.sample_weights.get_weights_by_return(
        labeling_info.reindex(X_train.index),
        data.loc[X_train.index, 'close_orig'],
        num_threads=1)
elif sample_weights_type == 'time_decay':
    sample_weigths = ml.sample_weights.get_weights_by_time_decay(
        labeling_info.reindex(X_train.index),
        data.loc[X_train.index, 'close_orig'],
        decay=0.5, num_threads=1)
elif labeling_technique is 'trend_scanning':
    sample_weigths = labeling_info['t_value'].reindex(X_train.index).abs()


### CROS VALIDATION STEPS
if cv_type == 'purged_kfold':
    cv = ml.cross_validation.PurgedKFold(
        n_splits=cv_number,
        samples_info_sets=labeling_info['t1'].reindex(X_train.index))


# MODEL
# convert pandas df to xgboost matrix
dmatrix_train = xgb.DMatrix(data=X_train, label=y_train.replace(-1, 0))
dmatrix_test = xgb.DMatrix(data=X_test, label=y_test.replace(-1, 0))

# parameters for GridSearch
parameters = {'max_depth': range(2, 6, 1),
              'n_estimators': range(50, 200, 50),
              'learning_rate': [0.10, 1, 0.05]
            }

# define estimator
estimator = xgb.XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=3
)

# define grid search
clf = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring='roc_auc',
    refit=True,
    cv=cv
)

# fit random search
clf.fit(
    X_train, y_train, verbose=True
    # early_stopping_rounds=20, eval_set=[X_val, y_val], eval_metric='auc'
        )
learning_rate, max_depth, n_estimators = clf.best_params_.values()

# model scores
clf_predictions = clf.predict(X_test)
clf_f1_score = sklearn.metrics.f1_score(y_test, clf_predictions)
print(f'f1_score: {clf_f1_score}')
print(f'optimal_model_depth: {depth}')
print(f'n_estimators: {n_estimators}')
print(f'max_features {n_features}')
