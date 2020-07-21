# # fundamental modules
# import numpy as np
# import scipy
# import pandas as pd
# from pathlib import Path
# import matplotlib.pyplot as plt
# from matplotlib import pyplot
# from IPython.display import display
# # preprocessing
# from sklearn.model_selection import train_test_split
# # modelling
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
# from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit
# from sklearn.base import clone
# import xgboost
# import shap
# import h2o
# from h2o.automl import H2OAutoML
# # metrics 
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     accuracy_score,
#     roc_curve,
#     log_loss
#     )
# # finance
# import mlfinlab as ml
# import mlfinlab.backtest_statistics as bs
# import pyfolio as pf
# import trademl
# # other
# from plotnine import *



# ### GLOBAL (CONFIGS)

# DATA_PATH = Path('C:/Users/Mislav/algoAItrader/data/spy_store_stat.h5')


# ### IMPORT AND ADD FEATURES

# with pd.HDFStore(DATA_PATH) as store:
#     spy = store.get('spy_store_stat')
# spy = spy.iloc[:1000000]


# ### SIMPLE CORRELATION ANALYSIS

# # # simple correlation matrix from pandas
# # remove_ohl = ['open', 'low', 'high', 'vixFirst', 'vixHigh', 'vixLow']  # correlatin > 0.99
# # spy.drop(columns=remove_ohl, inplace=True)  #correlated with close
# # corr_matrix = spy.drop(columns=('close_orig')).corr(method='pearson')  # kendall comp. int.

# # # remove varibles with correlation grater than 0.99
# # corr_matrix.head()


# ### TRIPLE-BARRIER LABELING

# # Compute daily volatility
# daily_vol = ml.util.get_daily_vol(close=spy['close_orig'], lookback=50)

# # Apply Symmetric CUSUM Filter and get timestamps for events
# cusum_events = ml.filters.cusum_filter(spy['close_orig'],
#                                         threshold=daily_vol.mean()*0.70)

# # Compute vertical barrier
# vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events,
#                                                         close=spy['close_orig'],
#                                                         num_days=4200)
    

# # make triple barriers (if side_prediction arg is omitted, return -1, 0, 1 
# # (w/t which touched first))
# pt_sl = [1.5, 1.5]  # IF ONLY SECONDARY (ML) MODEL HORIZONTAL BARRIERS SYMMETRIC!
# min_ret = 0.005
# triple_barrier_events = ml.labeling.get_events(
#     close=spy['close_orig'],
#     t_events=cusum_events,
#     pt_sl=pt_sl,
#     target=daily_vol,
#     min_ret=min_ret,
#     num_threads=1,
#     vertical_barrier_times=vertical_barriers)
# display(triple_barrier_events.head(10))

# # labels
# labels = ml.labeling.get_bins(triple_barrier_events, spy['close_orig'])
# display(labels.head(10))
# display(labels.bin.value_counts())
# labels = ml.labeling.drop_labels(labels)
# triple_barrier_events = triple_barrier_events.reindex(labels.index)


# ### PREPARE MOEL

# # get data at triple-barrier evetns
# X = spy.drop(columns=['close_orig']).reindex(labels.index)  # PROVJERITI OVO. MODA IPAK IDE TRIPPLE-BARRIER INDEX ???
# y = labels['bin']
# print('X shape: ', X.shape); print('y shape: ', y.shape)
# print('y counts:\n', y.value_counts())

# # train test split
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=0.10, 
#                                                     shuffle=False, 
#                                                     stratify=None)
# print(X_train.shape); print(y_train.shape)
# print(X_test.shape); print(y_test.shape)


# ### SAMPLE WEIGHTS

# # return weights
# return_sample_weights = ml.sample_weights.get_weights_by_return(
#     triple_barrier_events.loc[X_train.index],
#     spy.loc[X_train.index, 'close_orig'],
#     num_threads=1)
# # time decaz weights
# time_sample_weights = ml.sample_weights.get_weights_by_time_decay(
#     triple_barrier_events.loc[X_train.index],
#     spy.loc[X_train.index, 'close_orig'],
#     decay=0.5, num_threads=1)


# ### ML MODEL

# # RF GRID CV

# # parameters for GridSearch
# parameters = {'max_depth': [2, 3, 4, 5, 10],
#                 'n_estimators': [600, 1000, 1400]}

# # CV generators
# cv_gen_purged = ml.cross_validation.PurgedKFold(
#     n_splits=4,
#     samples_info_sets=triple_barrier_events.reindex(X_train.index).t1)
# rf = RandomForestClassifier(criterion='entropy',
#                             max_features=10,
#                             min_weight_fraction_leaf=0.05,
#                             class_weight='balanced_subsample')
# clf = GridSearchCV(rf,
#                     param_grid=parameters,
#                     scoring='f1',
#                     n_jobs=8,
#                     cv=cv_gen_purged)
# clf.fit(X_train, y_train, sample_weight=return_sample_weights)
# depth, n_estimators = clf.best_params_.values()
# rf_best = RandomForestClassifier(criterion='entropy',
#                                     max_features=1,
#                                     min_weight_fraction_leaf=0.05,
#                                     max_depth=depth,
#                                     n_estimators=n_estimators,
#                                     class_weight='balanced_subsample')
# rf_best.fit(X_train, y_train, sample_weight=return_sample_weights)
# trademl.modeling.metrics_summary.display_clf_metrics(
#     rf_best, X_train, X_test, y_train, y_test)
# trademl.modeling.metrics_summary.plot_roc_curve(
#     rf_best, X_train, X_test, y_train, y_test)


# ############### NOT ENOUGH MEMORY ##################

# # # PURGED BAGGING CLF

# # X_train_test = X_train.astype(np.float32)
# # X_train_test = X_train_test.iloc[:, :20]
# # # define parameters
# # parameters = {'max_depth': [2, 3, 4, 5, 7],
# #               'n_estimators': [100, 500, 1000]}

# # # CV splits
# # cv_gen_purged = ml.cross_validation.PurgedKFold(
# #     n_splits=4,
# #     samples_info_sets=triple_barrier_events.reindex(X_train.index).t1)

# # # grid search inside simple loop
# # max_cross_val_score = -np.inf
# # top_model = None
# # for m_depth in parameters['max_depth']:
# #     for n_est in parameters['n_estimators']:
# #         clf_base = DecisionTreeClassifier(criterion='entropy', random_state=42, 
# #                                         max_depth=m_depth, class_weight='balanced')
# #         clf = ml.ensemble.SequentiallyBootstrappedBaggingClassifier(
# #             samples_info_sets=triple_barrier_events.reindex(X_train_test.index).t1,
# #             price_bars = spy.loc[X_train_test.index.min():X_train_test.index.max(), 'close_orig'],
# #             n_estimators=n_est, base_estimator=clf_base,
# #             random_state=42, n_jobs=8, oob_score=False,
# #             max_features=1.)
# #         temp_score_base = ml.cross_validation.ml_cross_val_score(
# #             clf, X_train_test, y_train, cv_gen_purged, scoring='f1')
# #         if temp_score_base.mean() > max_cross_val_score:
# #             max_cross_val_score = temp_score_base.mean()
# #             print(temp_score_base.mean())
# #             top_model = clf

# ############## NOT ENOUGH MEMORY #################


# # H2O AUTO ML

# # import and init
# h2o.init(nthreads=16, max_mem_size=10)

# # convert X and y to h2o df
# # train = trademl.modeling.utils.cbind_pandas_h2o(X_train, y_train)
# # test = trademl.utils.cbind_pandas_h2o(X_test, y_test)
# train = cbind_pandas_h2o(X_train, y_train)
# test = cbind_pandas_h2o(X_test, y_test)

# # Identify response and predictor variables
# y = 'bin'
# x = list(train.columns)
# x.remove(y)  #remove the response

# # Automl train
# aml = H2OAutoML(max_models=12, seed=1)
# aml.train(x=x, y=y, training_frame=train)
# lb = aml.leaderboard

# # performance
# m = h2o.get_model(lb[0,"model_id"])
# predictions = m.predict(test)
# predictions['predict'].table()
# m.confusion_matrix()  # train set
# print(m.model_performance(train))  # train set
# print(m.model_performance(test))  # test set


# ### FEATURE IMPORTANCE

# # mean decreasing impurity
# mdi_feature_imp = ml.feature_importance.mean_decrease_impurity(
#     rf_best, X_train.columns)
# ml.feature_importance.plot_feature_importance(
#     mdi_feature_imp, 0, 0, save_fig=True,
#     output_path='mdi_feat_imp.png')

# # mean decreasing accuracy
# mda_feature_imp = ml.feature_importance.mean_decrease_accuracy(
#     rf_best, X_train, y_train, cv_gen_purged,
#     scoring=log_loss,
#     sample_weight_train=return_sample_weights.values)
# ml.feature_importance.plot_feature_importance(
#     mda_feature_imp, 0, 0, save_fig=True,
#     output_path='mda_feat_imp.png')

# # # single feature importance
# # rf_best_ = clone(rf_best)  # seems sfi change learner somehow, and can't use it later
# # sfi_feature_imp = ml.feature_importance.single_feature_importance(
# #     rf_best_, X_train, y_train, cv_gen_purged,
# #     scoring=accuracy_score,
# #     sample_weight_train=return_sample_weights.values)
# # ml.feature_importance.plot_feature_importance(
# #     sfi_feature_imp, 0, 0, save_fig=True,
# #     output_path='sfi_feat_imp.png')

# # Shapley valeus
# # load JS visualization code to notebook
# shap.initjs()
# # explain the model's predictions using SHAP
# explainer = shap.TreeExplainer(model=rf_best, model_output='raw')
# shap_values = explainer.shap_values(X_train)
# # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_train.iloc[0,:])
# # summarize feature importance
# shap.summary_plot(shap_values, X_train, plot_type='bar')

# # H2O FEATURE IMPORTANCE

# # feature importnce: https://stackoverflow.com/questions/51640086/is-it-possible-to-get-a-feature-importance-plot-from-a-h2o-automl-model
# m = h2o.get_model(lb[1,"model_id"])
# feature_importance = m.varimp(use_pandas=True)
# try:
#     m.varimp_plot()
# except TypeError as te:
#     print(te)

# # contributions
# contributions = m.predict_contributions(test)
# contributions_matrix = contributions.as_data_frame()  # Convert H2OFrame to use with SHAP
# shap_values_h2o = contributions_matrix.iloc[:,:-1]  # Calculate SHAP values for all features
# shap.summary_plot(shap_values, train.as_data_frame().drop(columns=['bin']), plot_type='bar')






# ### XGBOOST ALGORITHM

# # X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, 
# #                                                     shuffle=False, 
# #                                                     test_size=0.10,
# #                                                     stratify=None)

# # paramGrid = {"max_depth" : [2, 3],
# #              'n_estimators': [600, 1000]}

# # fit_params={"early_stopping_rounds":42, 
# #             "eval_metric" : "auc", 
# #             "eval_set" : [[X_val, y_val]]}

# # model = xgboost.XGBRFClassifier()

# # cv = TimeSeriesSplit(n_splits=4).get_n_splits([X_train, y_train])
# # gridsearch = GridSearchCV(model, paramGrid, verbose=1,             
# #                           cv=cv, n_jobs=8)
# # gridsearch.fit(X_train_, y_train_, **fit_params)

# # predictions = xgb_model.predict(X_test)
# # pd.value_counts(predictions)
# # trademl.modeling.metrics_summary.display_clf_metrics(
# #     xgb_model, X_train, X_test, y_train, y_test)

# # clf = GridSearchCV(rf,
# #                    param_grid=parameters,
# #                    scoring='f1',
# #                    n_jobs=8,
# #                    cv=cv_gen_purged)
# # clf.fit(X_train,y_train)
# # print(clf.best_score_)
# # print(clf.best_params_)
# # depth, n_estimators = clf.best_params_.values()
# # rf_best = RandomForestClassifier(criterion='entropy',
# #                                     max_features=1,
# #                                     min_weight_fraction_leaf=0.05,
# #                                     max_depth=depth,
# #                                     n_estimators=n_estimators,
# #                                     class_weight='balanced_subsample')
# # rf_best.fit(X_train, y_train, sample_weight=return_sample_weights)
# # trademl.modeling.metrics_summary.display_clf_metrics(
# #     rf_best, X_train, X_test, y_train, y_test)
# # trademl.modeling.metrics_summary.plot_roc_curve(
# #     rf_best, X_train, X_test, y_train, y_test)

# # # save model
# # # bst.save_model('0001.model')


# # cv_gen_purged = ml.cross_validation.PurgedKFold(
# #     n_splits=4,
# #     samples_info_sets=triple_barrier_events.reindex(X_train.index).t1)


# ### BACKTEST PLOTS


# def cumulative_returns(close, raw=True):
#     # perfrormance
#     if raw:
#         close = close.pct_change()
#     close = (1 + close).cumprod()
#     return close.dropna()


# def cumulative_returns_tb(trbar_info, predictions, bet_sizes=None, time_index=True):
    
#     return_adj = np.where(
#         predictions == trbar_info['bin'],
#         trbar_info['ret'].abs(), -(trbar_info['ret'].abs()))
#     if bet_sizes is not None:
#         return_adj = return_adj * bet_sizes
#     if time_index:
#         return_adj = pd.Series(return_adj, index=trbar_info.t1) #.to_frame()
#         return_adj.index.name = None
#     else:
#         return_adj = pd.Series(return_adj)
#     perf = cumulative_returns(return_adj, raw=False)
#     return perf.rename('cumulative_return', inplace=True)


# def minute_to_daily_return(trbar_info, predictions, bet_sizes=None):
#     return_adj = np.where(
#         predictions == trbar_info['bin'],
#         trbar_info['ret'].abs(), -(trbar_info['ret'].abs()))
#     if bet_sizes is not None:
#         return_adj = return_adj * bet_sizes
#     return_adj = pd.Series(return_adj, index=trbar_info.t1)
#     daily_returns = (1 + return_adj).resample('B').prod() - 1
#     return daily_returns


# # raw test set perfrormance
# time_range = pd.date_range(X_test.index[0], X_test.index[-1], freq='1Min')
# close = spy.close_orig.reindex(time_range).to_frame().dropna()
# close_cum = cumulative_returns(close)
# close_cum.plot()

# # tripple barrier returns
# tb_backtest_info = pd.concat([labels, triple_barrier_events.t1], axis=1)
# tb_backtest_info = tb_backtest_info.reindex(time_range).dropna()
# # random forest
# predictions_tb = rf_best.predict(X_test)
# pred_rf_crt = cumulative_returns_tb(tb_backtest_info, predictions_tb)
# pred_rf_cr = cumulative_returns_tb(tb_backtest_info, predictions_tb, time_index=False)
# # XGBoost
# pred_h2o_automl = m.predict(test)['predict'].as_data_frame().squeeze().values
# pred_h2o_crt = cumulative_returns_tb(tb_backtest_info, pred_h2o_automl)
# pred_h2o_cr = cumulative_returns_tb(tb_backtest_info, pred_h2o_automl, time_index=False)
# # plot
# pd.concat([pred_rf_crt, pred_h2o_crt], axis=1).plot()
# pd.concat([pred_rf_cr, pred_h2o_cr], axis=1).plot()
# all_together = close_cum.merge(pred_rf_crt, how='left', left_index=True, right_index=True)
# all_together = all_together.merge(pred_h2o_crt, how='left', left_index=True, right_index=True)
# all_together.plot()

# # tripple barrier returns with bet_size_budget
# bet_sizes = ml.bet_sizing.bet_size_budget(tb_backtest_info['t1'],
#                                           predictions_tb)
# cum_returns_rf = cumulative_returns_tb(tb_backtest_info, pred_h2o_automl,
#                                        bet_sizes['bet_size'].abs().values,
#                                        time_index=False)
# cum_returns_rft = cumulative_returns_tb(tb_backtest_info, pred_h2o_automl,
#                                         bet_sizes['bet_size'].abs().values)
# pred_h2o_bs = cumulative_returns_tb(tb_backtest_info, predictions_tb,
#                                     bet_sizes['bet_size'].abs().values,
#                                     time_index=False)
# pred_h2o_bst = cumulative_returns_tb(tb_backtest_info, predictions_tb,
#                                      bet_sizes['bet_size'].abs().values)
# # plot
# pd.concat([cum_returns_rft, pred_h2o_bst], axis=1).plot()
# pd.concat([cum_returns_rf, pred_h2o_bs], axis=1).plot()
# all_together = close_cum.merge(cum_returns_rft, how='left', left_index=True, right_index=True)
# all_together = all_together.merge(pred_h2o_bst, how='left', left_index=True, right_index=True)
# all_together.plot()


# # tripple barrier returns with bet_size_probability
# pred = pd.Series(rf_best.predict(X_test))
# prob = pd.Series(rf_best.predict_proba(X_test)[:, 0])
# bet_sizes =  ml.bet_sizing.bet_size_probability(
#     events=tb_backtest_info[['t1']],
#     prob=prob, pred=pred, num_classes=2,
#     step_size=0.01, num_threads=1)
# bet_sizes = bet_sizes.abs()
# cum_returns_rf = cumulative_returns_tb(tb_backtest_info, pred_h2o_automl,
#                                        bet_sizes.values,
#                                        time_index=False)
# cum_returns_rft = cumulative_returns_tb(tb_backtest_info, pred_h2o_automl,
#                                         bet_sizes.values)
# pred_h2o_bs = cumulative_returns_tb(tb_backtest_info, predictions_tb,
#                                     bet_sizes.values,
#                                     time_index=False)
# pred_h2o_bst = cumulative_returns_tb(tb_backtest_info, predictions_tb,
#                                      bet_sizes.values)
# # plot
# pd.concat([cum_returns_rft, pred_h2o_bst], axis=1).plot()
# pd.concat([cum_returns_rf, pred_h2o_bs], axis=1).plot()
# all_together = close_cum.merge(cum_returns_rft, how='left', left_index=True, right_index=True)
# all_together = all_together.merge(pred_h2o_bst, how='left', left_index=True, right_index=True)
# all_together.plot()


# ### BACKTEST STATISTICS

# # define arguments for backtest staticis functions (my and pyfolios)
# daily_returns = minute_to_daily_return(tb_backtest_info, pred_h2o_automl,
#                                  bet_sizes.values)
# daily_cum_returns = (1 + daily_returns).cumprod().resample('B').last().dropna()
# daily_returns_raw = spy['close_orig'].resample('B').last().dropna().pct_change().dropna()

# # Pyfolio
# perf_func = pf.timeseries.perf_stats
# perf_stats_all = perf_func(returns=daily_returns, 
#                            factor_returns=None)
# perf_stats_all = perf_func(returns=daily_returns_raw, 
#                            factor_returns=None)


# # my backtest statistics
# def backtest_stat(returns):
#     # RUNS
#     pos_concentr, neg_concentr, hour_concentr = bs.all_bets_concentration(returns, frequency='D')
#     drawdown, tuw = bs.drawdown_and_time_under_water(returns, dollars = False)
#     drawdown_dollars, _ = bs.drawdown_and_time_under_water(returns, dollars = True)
    
#     # EFFICIENCY
#     days_observed = (returns.index[-1] - returns.index[0]) / np.timedelta64(1, 'D')
#     cumulated_return = ((1 + returns).cumprod() - 1)[-1]
#     annual_return = (1 + cumulated_return)**(365/days_observed) - 1
    
#     # annualized sharpe ratio
#     annualized_sr = bs.sharpe_ratio(returns,
#                                     entries_per_year=252,
#                                     risk_free_rate=0)
    
#     # information ratio
#     trading_days = 252
#     benchmark = 0.02
#     daily_risk_free_ratio = (1 + benchmark)**(1/252) - 1
#     log_daily_risk_free_ratio = np.log(1 + daily_risk_free_ratio)
#     information_ratio = bs.information_ratio(returns,
#                                              log_daily_risk_free_ratio,
#                                              entries_per_year=trading_days)
    
#     # probabilistic sharpe ratio
#     probabilistic_sr =bs.probabilistic_sharpe_ratio(
#         observed_sr=annualized_sr,
#         benchmark_sr=1,
#         number_of_returns=days_observed,
#         skewness_of_returns=returns.skew(),
#         kurtosis_of_returns=returns.kurt())
    
#     # deflated sharpe ratio
#     deflated_sr = bs.deflated_sharpe_ratio(
#         observed_sr=annualized_sr,
#         sr_estimates=[0.5**(1/2), 100],
#         number_of_returns=days_observed,
#         skewness_of_returns=returns.skew(),
#         kurtosis_of_returns=returns.kurt(),
#         estimates_param=True)

#     # merge all statistics to dictionary
#     backtest_statistics = {
#         'Mean annual return': annual_return,
#         'Positive concetration': pos_concentr,
#         'Negative concetration': neg_concentr,
#         'Hour concetration': hour_concentr,
#         'The 95th percentile Drawdown': drawdown.quantile(.95),
#         'The 95th percentile Drawdown in dollars': drawdown_dollars.quantile(.95),
#         'The 95th percentile of Time under water': tuw.quantile(.95),
#         'Maximum Drawdown': drawdown.max(),
#         'Maximum Drawdown in dolars': drawdown_dollars.max(),
#         'Maximum Drawdown time': tuw.max(),
#         'Average return from positive bars': returns[returns>0].mean(),
#         'Counter from positive bars': returns[returns>0].count(),
#         'Average return from negative bars': returns[returns<0].mean(),
#         'Counter from negative bars': returns[returns<0].count(),
#         'Annualized Sharpe Ratio': annualized_sr,
#         'Information ratio (yearly risk-free rate 2%)': information_ratio,
#         'Probabilistic Sharpe Ratio with benchmark SR of 1': probabilistic_sr,
#         'Deflated Sharpe Ratio with 100 trails and 0.5 variance': deflated_sr
#     }
#     # dictionary to dataframe    
#     df = pd.DataFrame.from_dict(backtest_statistics, orient='index')

#     return df

# backtest_stat(daily_returns)



# ############# CHOOSE MOST IMPORTANT VARIABLES ###############


# # select most important variables
# mdi_feature_imp['mean'].nlargest(5)
# mda_feature_imp['mean'].nlargest(5)
# mda_feature_imp['mean'].nsmallest(5)
# shap_rf = ['WILLR12000', 'SAREXT', 'AROON_3000_aroondown', 'MFI2400', 'RSI300']


# pd.concat([pd.Series(shap_rf), pd.Series(X_train.columns)], axis=1)

# x = np.sort(np.mean(shap_values[0], axis=0))

# important_features
# spy_sample = spy[]
# shap_values_h2o


# from sklearn.base import BaseEstimator, TransformerMixin


# class TripleBarrier(BaseEstimator, TransformerMixin):
#     def __init__(self, lookback=50, scale_vol=1, num_days=4200): # no *args or **kargs
#         # parameters for triple-barrier
#         self.lookback = lookback
#         self.scale_vol = scale_vol
#         self.num_days = num_days
        
        
#     def fit(self, X, y=None):
#         # Compute daily volatility
#         daily_vol = ml.util.get_daily_vol(close=X['close_orig'], lookback=self.lookback)
#         # Apply Symmetric CUSUM Filter and get timestamps for events
#         cusum_events = ml.filters.cusum_filter(
#             X['close_orig'], threshold=daily_vol.mean()*self.scale_vol)
#         # Compute vertical barrier
#         vertical_barriers = ml.labeling.add_vertical_barrier(
#             t_events=cusum_events, close=X['close_orig'], num_days=self.num_days)
#         # make triple barriers (if side_prediction arg is omitted, return -1, 0, 1 
#         # (w/t which touched first))
#         pt_sl = [1.5, 1.5]  # IF ONLY SECONDARY (ML) MODEL HORIZONTAL BARRIERS SYMMETRIC!
#         triple_barrier_events = ml.labeling.get_events(
#             close=spy['close_orig'],
#             t_events=cusum_events,
#             pt_sl=pt_sl,
#             target=daily_vol,
#             min_ret=min_ret,
#             num_threads=1,
#             vertical_barrier_times=vertical_barriers)
#         display(triple_barrier_events.head(10))
#         return cusum_events # nothing else to do
    
    
#     def transform(self, X, y=None):
#         # labels
#         labels = ml.labeling.get_bins(triple_barrier_events, spy['close_orig'])
#         labels = ml.labeling.drop_labels(labels)
#         triple_barrier_events = triple_barrier_events.reindex(labels.index)
#         X = pd.concat([triple_barrier_events, X], axis=1, join='inner')
        
#         return X        


# tbpipe = TripleBarrier()
# spy_test = tbpipe.fit(spy)
# spy_test = tbpipe.transform(spy)
# spy_test.head()

