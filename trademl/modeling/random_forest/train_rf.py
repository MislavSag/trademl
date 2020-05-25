# fundamental modules
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
import joblib
import json
import sys
# preprocessing
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier
from sklearn.base import clone
import xgboost
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


### GLOBALS
DATA_PATH = 'D:/market_data/usa/ohlcv_features/'


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
data['close_orig'] = data['close']  # with original close reslts are pretty bad!


### NON-MODEL HYPERPARAMETERS
std_outlier = 10
tb_volatility_lookback = 50
tb_volatility_scaler = 1
tb_triplebar_num_days = 3
tb_triplebar_pt_sl = [1, 1]
tb_triplebar_min_ret = 0.003
sample_weights_type = 'returns'
cv_type = 'purged_kfold'
cv_number = 4
max_features = 15
max_depth = 2
rand_state = 3
n_estimators = 1000
remove_ind_with_high_period = True


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

### TRIPLE BARRIER LABELING
triple_barrier_pipe= tml.modeling.pipelines.TripleBarierLabeling(
    close_name='close_orig',
    volatility_lookback=tb_volatility_lookback,
    volatility_scaler=tb_volatility_scaler,
    triplebar_num_days=tb_triplebar_num_days,
    triplebar_pt_sl=tb_triplebar_pt_sl,
    triplebar_min_ret=tb_triplebar_min_ret,
    num_threads=1
)
tb_fit = triple_barrier_pipe.fit(data)
X = tb_fit.transform(data)


### TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X.drop(columns=['close_orig']), tb_fit.triple_barrier_info['bin'],
    test_size=0.10, shuffle=False, stratify=None)


### SAMPLE WEIGHTS (DECAY FACTOR CAN BE ADDED!)
if sample_weights_type == 'returns':
    sample_weigths = ml.sample_weights.get_weights_by_return(
        tb_fit.triple_barrier_info.reindex(X_train.index),
        data.loc[X_train.index, 'close_orig'],
        num_threads=1)
elif sample_weights_type == 'time_decay':
    sample_weigths = ml.sample_weights.get_weights_by_time_decay(
        tb_fit.triple_barrier_info.reindex(X_train.index),
        data.loc[X_train.index, 'close_orig'],
        decay=0.5, num_threads=1)


### CROS VALIDATION STEPS
if cv_type == 'purged_kfold':
    cv = ml.cross_validation.PurgedKFold(
        n_splits=cv_number,
        samples_info_sets=tb_fit.triple_barrier_info.reindex(X_train.index).t1)


# MODEL

# clf = joblib.load("rf_model.pkl")
clf = RandomForestClassifier(criterion='entropy',
                             max_features=max_features,
                             min_weight_fraction_leaf=0.05,
                             max_depth=max_depth,
                             n_estimators=n_estimators,
                             class_weight='balanced_subsample',
                             random_state=rand_state,
                             n_jobs=16)
# clf.fit(X_train, y_train, sample_weight=sample_weigths)
scores = ml.cross_validation.ml_cross_val_score(
    clf, X_train, y_train, cv_gen=cv, 
    sample_weight_train=sample_weigths,
    scoring=sklearn.metrics.f1_score)


### MODEL EVALUATION
# pogledati zero one loss: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.zero_one_loss.html
# mean, std and 95 percent intervals of scorers
print(f'mean_score: {scores.mean()}')
print(f'score_std: {scores.std()}')
print(f'confidence_intervals_95: {scores.std() * 2}')


# retrain the model if mean score is high enough (higher than 0.5)
if scores.mean() > 0.5:
    # refit the clf so I can calcute other metrics
    clf.fit(X_train, y_train, sample_weight=sample_weigths)
else:
    sys.exit("Bad performance!")

### CLF METRICS
tml.modeling.metrics_summary.clf_metrics(
    clf, X_train, X_test, y_train, y_test, avg='binary')  # HAVE TO FIX
tml.modeling.metrics_summary.plot_roc_curve(
    clf, X_train, X_test, y_train, y_test)


### FEATURE SELECTION
def feature_importance(clf, X_train, y_train):

    # clone clf to not change it
    clf_ = sklearn.clone(clf)
    clf_.fit(X_train, y_train)

    # SHAPE values
    explainer = shap.TreeExplainer(model=clf_, model_output='raw')
    shap_values = explainer.shap_values(X_train)

    return shap_values


def feature_importnace_vec(shap_val):
    # SHAP values
    vals= np.abs(shap_val).mean(0)
    feature_importance = pd.DataFrame(
        list(zip(X_train.columns, sum(vals))),
        columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(
        by=['feature_importance_vals'], ascending=False, inplace=True)
    return feature_importance


def plot_feature_importance(shap_val):
    # SHAP values
    shap.initjs()
    shap.summary_plot(shap_val, X_train, plot_type='bar', max_display=25)


fmp = tml.modeling.feature_importance.feature_importance(clf, X_train, y_train)
fpm = feature_importance(clf, X_train, y_train)
fpmv = feature_importnace_vec(shap_val=fpm)
plot_feature_importance(shap_val=fpm)


### REFIT THE MODEL WITH MOST IMPORTANT FEATURES
# X_train_important = X_train[fpmv['col_name'].head(25)]
# X_test_important = X_test[fpmv['col_name'].head(25)]
# clf_important = clf.fit(X_train_important, y_train)
# tml.modeling.metrics_summary.clf_metrics(
#     clf, X_train, X_test, y_train, y_test, avg='binary')  # HAVE TO FIX
# tml.modeling.metrics_summary.plot_roc_curve(
#     clf_important, X_train_important, X_test_important, y_train, y_test)


### SAVE THE MODEL AND FEATURES
joblib.dump(clf, "rf_model.pkl")
pd.Series(X_train.columns).to_csv('feature_names.csv', sep=',')
serialized_model = serialize_random_forest(clf)
with open('rf_model.json', 'w') as f:
    json.dump(serialized_model, f)


### BACKTESTING (RADI)

# BUY-SELL BACKTESTING STRATEGY
# true close 
time_range = pd.date_range(X_test.index[0], X_test.index[-1], freq='1Min')
close = data.close_orig.reindex(time_range).to_frame().dropna()
# predictions on test set
predictions = pd.Series(clf.predict(X_test), index=X_test.index)
# plot cumulative returns
hold_cash = tml.modeling.backtest.hold_cash_backtest(close, predictions)
hold_cash[['close_orig', 'cum_return']].plot()


### BACKTEST STATISTICS 

# def pyfolio_sheet(returns):
#     daily_returns = returns.resample('D').mean().dropna()
#     perf_func = pf.timeseries.perf_stats
#     perf_stats_all = perf_func(returns=daily_returns, 
#                                factor_returns=None)
#     return perf_stats_all

# strategy_pf = pyfolio_sheet(hold_cash['return'])
# bencha_pf = pyfolio_sheet(data.close_orig.resample('D').last().
#                             dropna().pct_change())
# pf_sheet = pd.concat([bencha_pf.rename('banchmark'),
#                       strategy_pf.rename('strategy')], axis=1)




# import  mlfinlab.backtest_statistics as bs
# def backtest_stat(returns):
#     # RUNS
#     pos_concentr, neg_concentr, hour_concentr = bs.all_bets_concentration(returns, frequency='min')
#     drawdown, tuw = bs.drawdown_and_time_under_water(returns, dollars = False)
#     drawdown_dollars, _ = bs.drawdown_and_time_under_water(returns, dollars = True)
    
#     # EFFICIENCY
#     days_observed = (price_series.index[-1] - price_series.index[0]) / np.timedelta64(1, 'D')
#     cumulated_return = price_series[-1]/price_series[0]
#     annual_return = (cumulated_return)**(365/days_observed) - 1
#     print('Annualized average return from the portfolio is' , annual_return)

#     # merge all statistics to dictionary
#     backtest_statistics = {
#         'Positive concetration': pos_concentr,
#         'Negative concetration': neg_concentr,
#         'Hour concetration': hour_concentr,
#         'The 95th percentile Drawdown': drawdown.quantile(.95),
#         'The 95th percentile Drawdown in dollars': drawdown_dollars.quantile(.95),
#         'The 95th percentile of Time under water': tuw.quantile(.95),
#         'Maximum Drawdown': drawdown.max(),
#         'Maximum Drawdown in dolars': drawdown_dollars.max(),
#         'Maximum Drawdown time': tuw.max()
#     }
#     # dictionary to dataframe    
#     df = pd.DataFrame.from_dict(backtest_statistics, orient='index')

#     return df


# returns = hold_cash['return'].dropna()
# price_series = hold_cash['adjusted_close'].dropna()
# backtest_stat(returns)
