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
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier
from sklearn.base import clone
import xgboost
import h2o
from h2o.automl import H2OAutoML
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
# finance packages
import trademl as tml
# import vectorbt as vbt


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
ts_look_forward_window = 2400  # 60 * 8 * 10 (10 days)
ts_min_sample_length = 30
ts_step = 5
tb_min_pct = 0.10
sample_weights_type = 'returns'
cv_type = 'purged_kfold'
cv_number = 4
rand_state = 3
stationary_close_lables = False
multiclass = True

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
    if multiclass:
        labeling_info['bin'] = tml.modeling.utils.balance_multiclass(labeling_info['t_value'])
elif labeling_technique == 'fixed_horizon':
    X = data.copy()
    labeling_info = ml.labeling.fixed_time_horizon(data['close_orig'], 
                                                   threshold=0.005, resample_by='B').dropna().to_frame()
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


### H2O AUTO ML

# import and init
h2o.init(nthreads=16, max_mem_size=10)

# convert X and y to h2o df
# train = pd.concat([X_train, sample_weigths.rename('sample_weigts')], axis=1)
train = tml.modeling.utils.cbind_pandas_h2o(X_train, y_train)  # X_train default
train['bin'] = train['bin'].asfactor()
test = tml.modeling.utils.cbind_pandas_h2o(X_test, y_test)
test['bin'] = test['bin'].asfactor()

# Identify response and predictor variables
y = 'bin'
x = list(train.columns)
x.remove(y)  #remove the response

# Automl train
aml = H2OAutoML(max_models=15,
                seed=3,
                balance_classes=True,
                sort_metric='mean_per_class_error',
                stopping_metric='mean_per_class_error')  #
aml.train(x=x, y=y, training_frame=train)  # weights_column='sample_weigts'
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

# performance
m = h2o.get_model(lb[0,"model_id"])
predictions_h2o = m.predict(test)
predictions_h2o['predict'].table()
m.confusion_matrix()  # train set
print(m.model_performance(train))  # train set
print(m.model_performance(test))  # test set

# feature importance
m = h2o.get_model(lb[2,"model_id"])
feature_importance = m.varimp(use_pandas=True)
try:
    m.varimp_plot()
except TypeError as te:
    print(te)
shap.initjs()
contributions = m.predict_contributions(test)
contributions_matrix = contributions.as_data_frame()
shap_values_h2o = contributions_matrix.iloc[:,:-1]
shap.summary_plot(shap_values_h2o, train.as_data_frame().drop(columns=['bin']), plot_type='bar', max_display=25)
