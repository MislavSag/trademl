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
from trademl.modeling.utils import time_method
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
# data['close_orig'] = data['close']  # with original close reslts are pretty bad!


### NON-MODEL HYPERPARAMETERS
labeling_technique = 'tripple_barrier'
std_outlier = 10
tb_volatility_lookback = 50
tb_volatility_scaler = 1
tb_triplebar_num_days = 30
tb_triplebar_pt_sl = [1, 1]
tb_triplebar_min_ret = 0.005
ts_look_forward_window = 4800  # 60 * 8 * 10 (10 days)
ts_min_sample_length = 30
ts_step = 5
tb_min_pct = 0.10
sample_weights_type = 'returns'
cv_type = 'purged_kfold'
cv_number = 4
rand_state = 3
n_estimators = 1000
remove_ind_with_high_period = True

### MODEL HYPERPARAMETERS
max_depth=3
max_features = 15

### POSTMODEL PARAMETERS
keep_important_features = 25
vectorbt_slippage = 0.0015
vectorbt_fees = 0.0015


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
if labeling_technique == 'tripple_barrier':
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
    X = tb_fit.transform(data)
elif labeling_technique == 'trend_scanning':
    trend_scanning_pipe = tml.modeling.pipelines.TrendScanning(
        close_name='close',
        volatility_lookback=tb_volatility_lookback,
        volatility_scaler=tb_volatility_scaler,
        ts_look_forward_window=ts_look_forward_window,
        ts_min_sample_length=ts_min_sample_length,
        ts_step=ts_step
        )
    ts_fit = trend_scanning_pipe.fit(data)
    X = trend_scanning_pipe.transform(data)


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
    scoring=sklearn.metrics.balanced_accuracy_score)  #sklearn.metrics.f1_score(average='weighted')


### MODEL EVALUATION
# pogledati zero one loss: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.zero_one_loss.html
# mean, std and 95 percent intervals of scorers
print(f'mean_score: {scores.mean()}')
print(f'score_std: {scores.std()}')
print(f'confidence_intervals_95: {scores.std() * 2}')


# retrain the model if mean score is high enough (higher than 0.5)
if scores.mean() < 0.55:
    print('Bad performance')
    
else:
    clf.fit(X_train, y_train, sample_weight=sample_weigths)

    ### CLF METRICS
    tml.modeling.metrics_summary.clf_metrics(
    clf, X_train, X_test, y_train, y_test, avg='binary')  # HAVE TO FIX
    # tml.modeling.metrics_summary.plot_roc_curve(
    # clf, X_train, X_test, y_train, y_test)


    ### FEATURE SELECTION
    fival = tml.modeling.feature_importance.feature_importance_values(
    clf, X_train, y_train)
    fivec = tml.modeling.feature_importance.feature_importnace_vec(
    fival, X_train)
    tml.modeling.feature_importance.plot_feature_importance(fival, X_train)


    ### REFIT THE MODEL WITH MOST IMPORTANT FEATURES
    X_train_important = X_train[
    fivec['col_name'].
    head(keep_important_features)].drop(columns=['STOCHRSI_96000_fastk'])
    X_test_important = X_test[
    fivec['col_name'].
    head(keep_important_features)].drop(columns=['STOCHRSI_96000_fastk'])
    clf_important = clf.fit(X_train_important, y_train)
    tml.modeling.metrics_summary.clf_metrics(
    clf_important, X_train_important,
    X_test_important, y_train, y_test, avg='binary', prefix='fi_')
    tml.modeling.metrics_summary.plot_roc_curve(
    clf_important, X_train_important, X_test_important,
    y_train, y_test, suffix=' with importnat features')


    ### BACKTESTING (RADI)

    # BUY-SELL BACKTESTING STRATEGY
    # true close 
    time_range = pd.date_range(X_test.index[0], X_test.index[-1], freq='1Min')
    close = data.close_orig.reindex(time_range).to_frame().dropna()
    # predictions on test set
    predictions = pd.Series(clf.predict(X_test_important), index=X_test_important.index)
    # plot cumulative returns
    hold_cash = tml.modeling.backtest.hold_cash_backtest(close, predictions)
    hold_cash[['close_orig', 'cum_return']].plot()

    # VECTORBT
    positions = pd.concat([close, predictions.rename('position')], axis=1)
    positions = tml.modeling.backtest.enter_positions(positions.values)
    positions = pd.DataFrame(positions, index=close.index, columns=['close', 'position'])
    entries = (positions[['position']] == 1).vbt.signals.first() # buy at first 1
    exits = (positions[['position']] == -1).vbt.signals.first() # sell at first 0
    portfolio = vbt.Portfolio.from_signals(close, entries, exits,
                                        slippage=vectorbt_slippage,
                                        fees=vectorbt_fees)
    print(f'vectorbt_total_return: {portfolio.total_return}')

    #TRIPLE-BARRIER BACKTEST
    tbpred = tb_fit.triple_barrier_info.loc[predictions.index]
    tbpred['ret_adj'] = np.where(tbpred['bin']==predictions, np.abs(tbpred['ret']), -np.abs(tbpred['ret']))
    total_return = (1 + tbpred['ret_adj']).cumprod().iloc[-1]
    print(f'tb_return_nofees_noslippage: {total_return}')


    ### SAVE THE MODEL AND FEATURES
    # joblib.dump(clf, "rf_model_25.pkl")
    # pd.Series(X_train_important.columns).to_csv('feature_names_25.csv', sep=',')
    # serialized_model = tml.modeling.utils.serialize_random_forest(clf)
    # with open('rf_model_25.json', 'w') as f:
    #     json.dump(serialized_model, f)


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


    ############## TEST
    # model_features = pd.Series(X_train.columns)
    # min_d = pd.read_csv('min_d.csv', sep=';', names=['feature', 'value'])
    # min_d = min_d[1:]
    # min_d_close = min_d.loc[(min_d['feature'] == 'close') | (min_d['feature'] == 'open'), ['feature', 'value']]
    # min_d_close.set_index(min_d_close['feature'], inplace=True)
    # min_d_close = min_d_close['value']
    # min_d.set_index(min_d['feature'], inplace=True)

    # tripple barrier vector vs backtest
    # tb_fit.triple_barrier_info
    # tb_fit.triple_barrier_info.loc['2019-01-01 00:00:00':]
    # tb_fit.triple_barrier_info.loc['2016-07-07']
    # tb_fit.triple_barrier_info.loc['2016-07-07 00:00:00':].shape
    # 1000000 / 200
    # costs_per_transaction = (1000000 / 200) * 0.05
    # costs_per_transaction * tb_fit.triple_barrier_info.loc['2016-07-07 00:00:00':].shape[0]

    # # test multiplie orders
    # data.close_orig
    # test = ml.util.get_daily_vol(data.close_orig, lookback=50)
    # test[tb_fit.triple_barrier_info.index]

    # # extract close series
    # close_test = data.close_orig

    # # Compute volatility
    # daily_vol_test = ml.util.get_daily_vol(close_test, lookback=50)

    # # Apply Symmetric CUSUM Filter and get timestamps for events
    # cusum_events_test = ml.filters.cusum_filter(close_test,
    #     threshold=daily_vol_test.mean()*1)

    # # Compute vertical barrier
    # vertical_barriers_test = ml.labeling.add_vertical_barrier(
    #     t_events=cusum_events_test,
    #     close=close_test,
    #     num_days=2) 

    # # tripple barier events
    # triple_barrier_events_test = ml.labeling.get_events(
    #     close=close_test,
    #     t_events=cusum_events_test,
    #     pt_sl=[1, 1],
    #     target=daily_vol_test,
    #     min_ret=0.01,
    #     num_threads=1,
    #     vertical_barrier_times=vertical_barriers)

    # # labels
    # labels = ml.labeling.get_bins(triple_barrier_events, close)
    # labels = ml.labeling.drop_labels(labels)
    ############## TEST
