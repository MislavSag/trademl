import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pycaret.classification import *
import sklearn
from sklearn.model_selection import train_test_split
import mlfinlab as ml
import trademl as tml


### DON'T SHOW GRAPH OPTION
# matplotlib.use("Agg")


### GLOBALS
DATA_PATH = 'D:/market_data/usa/ohlcv_features'

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
    labeling_info = ml.labeling.fixed_time_horizon(
        data['close_orig'], threshold=0.005, resample_by='B').dropna().to_frame()
    labeling_info = labeling_info.rename(columns={'close_orig': 'bin'})
    print(labeling_info.iloc[:, 0].value_counts())
    X = X.iloc[:-1, :]


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X.drop(columns=['close_orig']), labeling_info['bin'],
    test_size=0.10, shuffle=False, stratify=None)
train = pd.concat([X_train, y_train], axis=1)
train['bin'] = np.where(train['bin'] == -1, 0, train['bin'])
train['bin'] = train['bin'].astype(np.int)
test = pd.concat([X_test, y_test], axis=1)
test['bin'] = np.where(test['bin'] == -1, 0, test['bin'])
test['bin'] = test['bin'].astype(np.int)


# PYCARET SETUP

#intialize the setup
# numeric_features = [col for col in data.columns if col != ['tick_rule']]
exp_clf = setup(train,
                target='bin',
                train_size=0.9,
                categorical_features=['tick_rule', 'HT_TRENDMODE', 'chow_segment'],
                pca=True,
                pca_method='linear',
                pca_components=10,
                remove_multicollinearity=False,
                # outliers_threshold=None,
                silent=True,  # don't need to confrim data types
                html=False)
compare_models(fold=10, turbo=True)

# create models
et = create_model('et')
catboost = create_model('catboost')
lightgbm = create_model('lightgbm')
rf = create_model('rf')
dt = create_model('dt')
gbc = create_model('gbc')
xgboost = create_model('xgboost')
knn = create_model('knn')

# tune models
tuned_et = tune_model(et)
tuned_catboost = tune_model(catboost)
tuned_lightgbm = tune_model(lightgbm)
tunrd_rf = tune_model(rf)
tuned_dt = tune_model(dt)
tuned_gbc = tune_model(gbc)
tuned_xgboost = tune_model(xgboost)
tuned_knn = tune_model(knn)

# blend trained models
blend_specific = blend_models(
    estimator_list = [tuned_et, tuned_lightgbm, tunrd_rf, tuned_dt, tuned_gbc, tuned_xgboost, tuned_knn]
    )

# stack trained models
stacked_models = stack_models(
    estimator_list = [tuned_et, tuned_lightgbm, tunrd_rf, tuned_dt, tuned_gbc, tuned_xgboost, tuned_knn]
    )


# plot models
# plot_model(tuned_et, plot = 'auc')
# plot_model(tuned_et, plot = 'pr')
# plot_model(tuned_et, plot = 'feature')
# plot_model(tuned_et, plot = 'confusion_matrix')

# predictions
predict_model(tuned_et)
predict_model(tuned_catboost)
predict_model(tuned_lightgbm)
predict_model(tunrd_rf)
predict_model(tuned_dt)
predict_model(tuned_gbc)
predict_model(tuned_xgboost)
predict_model(tuned_knn)
predict_model(blend_specific)

# finalize the model
final_et = finalize_model(tuned_et)
final_catboost = finalize_model(tuned_catboost)
final_lightgbm = finalize_model(tuned_lightgbm)
final_rf = finalize_model(tunrd_rf)
final_dt = finalize_model(tuned_dt)
final_gbc = finalize_model(tuned_gbc)
final_xgboost = finalize_model(tuned_xgboost)
final_knn = finalize_model(tuned_knn)
final_blend_specific = finalize_model(blend_specific)

# predict on unseen dataset
unseen_predictions_et = predict_model(final_et, data=test)
unseen_predictions_et.head()

# predict on unseen dataset
predictions_et = predict_model(final_et, data=test)
predictions_catboost = predict_model(final_catboost, data=test)
predictions_lightgbm = predict_model(final_lightgbm, data=test)
predictions_rf = predict_model(final_rf, data=test)
predictions_dt = predict_model(final_dt, data=test)
predictions_gbc = predict_model(final_gbc, data=test)
predictions_xgboost = predict_model(final_xgboost, data=test)
predictions_knn = predict_model(final_knn, data=test)
predictions_blend = predict_model(final_blend_specific, data=test)

# accuracy scores
sklearn.metrics.accuracy_score(unseen_predictions_et['bin'], unseen_predictions_et['Label'])
sklearn.metrics.accuracy_score(predictions_catboost['bin'], predictions_catboost['Label'])
sklearn.metrics.accuracy_score(predictions_lightgbm['bin'], predictions_lightgbm['Label'])
sklearn.metrics.accuracy_score(predictions_rf['bin'], predictions_rf['Label'])
sklearn.metrics.accuracy_score(predictions_dt['bin'], predictions_dt['Label'])
sklearn.metrics.accuracy_score(predictions_gbc['bin'], predictions_gbc['Label'])
sklearn.metrics.accuracy_score(predictions_xgboost['bin'], predictions_xgboost['Label'])
sklearn.metrics.accuracy_score(predictions_knn['bin'], predictions_knn['Label'])
sklearn.metrics.accuracy_score(predictions_blend['bin'], predictions_blend['Label'])

# save models
save_model(final_et,'final_et_model')
save_model(final_catboost,'final_catboost_model')
save_model(final_lightgbm,'final_lightgbm_model')
save_model(final_rf,'final_rf_model')
save_model(final_dt,'final_dt_model')
save_model(final_gbc,'final_gbc_model')
save_model(final_xgboost,'final_xgboost_model')
save_model(final_knn,'final_knn_model')
save_model(final_blend_specific,'final_blend_model')

# load models
final_et = load_model('final_et_model')
final_catboost = load_model('final_catboost_model')
final_lightgbm = load_model('final_lightgbm_model')
final_rf = load_model('final_rf_model')
final_dt = load_model('final_dt_model')
final_gbc = load_model('final_gbc_model')
final_xgboost = load_model('final_xgboost_model')
final_knn = load_model('final_knn_model')
final_blend_specific = load_model('final_blend_model')
