# # fundamental modules
# import numpy as np
# from numba import njit
# import scipy
# import pandas as pd
# from pathlib import Path
# import matplotlib.pyplot as plt
# from matplotlib import pyplot
# import sklearn
# from IPython.display import display
# import time
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
# import trademl as tml
# # other
# from plotnine import *


# @njit
# def calculate_t_values(subset, min_sample_length, step):
#     """
#     For loop for calculating linear regression every n steps.
    
#     :param subset: (np.array) subset of indecies for which we want to calculate t values
#     :return: (float) maximum t value and index of maximum t value
#     """
#     max_abs_t_value = -np.inf  # Maximum abs t-value of b_1 coefficient among l values
#     max_t_value_index = None  # Index with maximum t-value
    
#     for forward_window in np.arange(min_sample_length, subset.shape[0], step):

#         y_subset = subset[:forward_window].reshape(-1, 1)  # y{t}:y_{t+l}

#         # Array of [1, 0], [1, 1], [1, 2], ... [1, l] # b_0, b_1 coefficients
#         X_subset = np.ones((y_subset.shape[0], 2))
#         X_subset[:, 1] = np.arange(y_subset.shape[0])

#         # Get regression coefficients estimates
#         xy = X_subset.transpose() @ y_subset
#         xx = X_subset.transpose() @ X_subset

#         #   check for singularity
#         det = np.linalg.det(xx)
        
#         # get coefficient and std from linear regression
#         if det == 0:
#             b_mean = [np.nan]
#             b_std = [[np.nan, np.nan]]
#         else:
#             xx_inv = np.linalg.inv(xx)
#             b_mean = xx_inv @ xy
#             err = y_subset - (X_subset @ b_mean)
#             b_std = np.dot(np.transpose(err), err) / (X_subset.shape[0] - X_subset.shape[1]) * xx_inv
        
#         # Check if l gives the maximum t-value among all values {0...L}
#             t_beta_1 = (b_mean[1] / np.sqrt(b_std[1, 1]))[0]
#             if abs(t_beta_1) > max_abs_t_value:
#                 max_abs_t_value = abs(t_beta_1)
#                 max_t_value = t_beta_1
#                 max_t_value_index = forward_window
                
#     return max_t_value_index, max_t_value


# def my_trend_scanning_labels(price_series: pd.Series, t_events: list = None, look_forward_window: int = 20,
#                           min_sample_length: int = 5, step: int = 1) -> pd.DataFrame:
#     """
#     `Trend scanning <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419>`_ is both a classification and
#     regression labeling technique.

#     That can be used in the following ways:

#     1. Classification: By taking the sign of t-value for a given observation we can set {-1, 1} labels to define the
#        trends as either downward or upward.
#     2. Classification: By adding a minimum t-value threshold you can generate {-1, 0, 1} labels for downward, no-trend,
#        upward.
#     3. The t-values can be used as sample weights in classification problems.
#     4. Regression: The t-values can be used in a regression setting to determine the magnitude of the trend.

#     The output of this algorithm is a DataFrame with t1 (time stamp for the farthest observation), t-value, returns for
#     the trend, and bin.

#     :param price_series: (pd.Series) close prices used to label the data set
#     :param t_events: (list) of filtered events, array of pd.Timestamps
#     :param look_forward_window: (int) maximum look forward window used to get the trend value
#     :param min_sample_length: (int) minimum sample length used to fit regression
#     :param step: (int) optimal t-value index is searched every 'step' indices
#     :return: (pd.DataFrame) of t1, t-value, ret, bin (label information). t1 - label endtime, tvalue,
#         ret - price change %, bin - label value based on price change sign
#     """
#     # pylint: disable=invalid-name

#     if t_events is None:
#         t_events = price_series.index

#     t1_array = []  # Array of label end times
#     t_values_array = []  # Array of trend t-values

#     for index in t_events:
#         subset = price_series.loc[index:].iloc[:look_forward_window]  # Take t:t+L window
#         if subset.shape[0] >= look_forward_window:

#             # linear regressoin for every index
#             max_t_value_index, max_t_value = calculate_t_values(subset.values,
#                                                                 min_sample_length,
#                                                                 step)

#             # Store label information (t1, return)
#             label_endtime_index = subset.index[max_t_value_index - 1]
#             t1_array.append(label_endtime_index)
#             t_values_array.append(max_t_value)

#         else:
#             t1_array.append(None)
#             t_values_array.append(None)

#     labels = pd.DataFrame({'t1': t1_array, 't_value': t_values_array}, index=t_events)
#     labels.loc[:, 'ret'] = price_series.reindex(labels.t1).values / price_series.reindex(labels.index).values - 1
#     labels['bin'] = np.sign(labels.t_value)

#     return labels


# @njit
# def my__get_dfc_for_t(molecule_range, series_lag_values_start, series_diff, series_lag_values):
#     """
#     Multithread implementation of Chow-Type Dickey-Fuller Test, p.251-252
#     :param series: (pd.Series) series to test
#     :param min_length: (int) minimum sample length used to estimate statistics
#     :param num_threads: (int): number of cores to use
#     :return: (pd.Series) of Chow-Type Dickey-Fuller Test statistics
#     """
#     dfc_series = []
#     for i in molecule_range:
#         ### TEST
#         # index = molecule[0]
#         ### TEST
#         series_lag_values_ = series_lag_values.copy()
#         series_lag_values_[:(series_lag_values_start + i)] = 0  # D_t* indicator: before t* D_t* = 0

#         # define x and y for regression
#         y = series_diff
#         x = series_lag_values_.reshape(-1, 1)
        
#         # Get regression coefficients estimates
#         xy = x.transpose() @ y
#         xx = x.transpose() @ x

#         # calculate to check for singularity
#         det = np.linalg.det(xx)

#         # get coefficient and std from linear regression
#         if det == 0:
#             b_mean = [np.nan]
#             b_std = [[np.nan, np.nan]]
#         else:
#             xx_inv = np.linalg.inv(xx)
#             coefs = xx_inv @ xy
#             err = y - (x @ coefs)
#             coef_vars = np.dot(np.transpose(err), err) / (x.shape[0] - x.shape[1]) * xx_inv
            
#         b_estimate, b_var = coefs[0], coef_vars[0][0]
#         dfc_series.append(b_estimate / (b_var ** 0.5))
        
#     return dfc_series


# def my_get_chow_type_stat(series: pd.Series, min_length: int = 20) -> pd.Series:
#     """
#     Multithread implementation of Chow-Type Dickey-Fuller Test, p.251-252
#     :param series: (pd.Series) series to test
#     :param min_length: (int) minimum sample length used to estimate statistics
#     :param num_threads: (int): number of cores to use
#     :return: (pd.Series) of Chow-Type Dickey-Fuller Test statistics
#     """
#     # Indices to test. We drop min_length first and last values
#     molecule = series.index[min_length:series.shape[0] - min_length]
#     molecule = molecule.values
#     molecule_range = np.arange(0, len(molecule))

#     series_diff = series.diff().dropna()
#     series_diff = series_diff.values
#     series_lag = series.shift(1).dropna()
#     series_lag_values = series_lag.values
#     series_lag_times_ = series_lag.index.values
#     series_lag_values_start = np.where(series_lag_times_ == molecule[0])[0].item() + 1
    
#     dfc_series = _get_dfc_for_t(molecule_range, series_lag_values_start, series_diff, series_lag_values)
    
#     dfc_series = pd.Series(dfc_series, index=molecule)
    
#     return dfc_series



# ### GLOBAL (CONFIGS)

# DATA_PATH = Path('C:/Users/Mislav/algoAItrader/data/spy_with_vix.h5')


# ### IMPORT AND ADD FEATURES

# with pd.HDFStore(DATA_PATH) as store:
#     spy = store.get('spy_with_vix')
# spy.sort_index(inplace=True)
# display(spy.head())
# display(spy.tail())


# ### REMOVE HIGHLY CORRELATED PREDICTORS

# # remove ohlc and calculate corr
# spy.drop(columns=['open', 'low', 'high', 'vixFirst', 'vixHigh', 'vixLow',
#                   'open_orig', 'high_orig', 'low_orig'], inplace=True)
# corrs = spy.drop(columns=['close_orig']).corr()
# corrs.head()
# corrs['DEMA5'].sort_values(ascending=False).head(100)


# ### SAMPLING

# # Compute daily volatility
# daily_vol = ml.util.get_daily_vol(close=spy['close_orig'], lookback=60)

# # Apply Symmetric CUSUM Filter and get timestamps for events
# cusum_events = ml.filters.cusum_filter(spy['close_orig'],
#                                        threshold=daily_vol.mean()*1)

# # keep X where CUSUM show time to trade
# spy_sample = spy.reindex(cusum_events)
# print(f'NUMBER OF CUSUM EVENTS:  {spy_sample.shape[0]}')

# ### LABELING

# # trend scanning on cusum events
# start = time.time()
# my_labels = my_trend_scanning_labels(
#     price_series=spy['close_orig'],
#     t_events=spy_sample.index,
#     look_forward_window=4800,  # 60 * 8 * 10 (10 days)
#     min_sample_length=30,
#     step=5
# )
# end = time.time()
# print("Elapsed (with compilation) = %s" % (end - start))

# # remove nan from matrix
# labels_clean = my_labels.dropna()

# # check balance of labels
# labels_clean.bin.value_counts()


# ### PREPARE MODEL

# # experimenting. execute igf want remove correlated assets
# corrs_ = corr.copy()
# step = 0
# while step <= corrs_.shape[0]:
#     print(step)
#     try: 
#         col = corrs_.iloc[:, step]
#         keep = ((col == 1) | (col == -1) | ((col<0.90) & (col>-0.90)))
#         corr_ = corrs_.loc[keep, keep]
#     except IndexError:
#         pass
#     step += 1

# spy_sample = pd.concat([spy.loc[:, corrs_.columns], spy['close_orig']], axis=1)


# # define X and y sets
# X = spy.drop(columns=['close_orig']).reindex(labels_clean.index)
# y = labels_clean['bin']
# print('X shape: ', X.shape); print('y shape: ', y.shape)
# print('y counts:\n', y.value_counts())

# # train test split
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=0.10, 
#                                                     shuffle=False, 
#                                                     stratify=None)
# print(X_train.shape); print(y_train.shape)
# print(X_test.shape); print(y_test.shape)

# # matthews scoring
# matthews = sklearn.metrics.make_scorer(sklearn.metrics.matthews_corrcoef)


# ### RANDOM FOREST MODEL

# # parameters for GridSearch
# parameters = {'max_depth': [2, 3, 4],
#               'n_estimators': [500, 800],
#               'max_features': [10, 20]
#              }

# # CV generators
# cv_gen_purged = ml.cross_validation.PurgedKFold(
#     n_splits=4,
#     samples_info_sets=labels_clean.reindex(X_train.index).t1)  # X_train.t1  labels.reindex(X_train).t1
# rf = RandomForestClassifier(criterion='entropy',
#                             min_weight_fraction_leaf=0.05,
#                             class_weight='balanced')
# clf = GridSearchCV(rf,
#                    param_grid=parameters,
#                    scoring='f1',
#                    n_jobs=12,
#                    cv=cv_gen_purged)
# clf.fit(X_train, y_train)  # , sample_weight=labels_clean.reindex(X_train.index).t_value.abs()
# depth, n_features, n_estimators = clf.best_params_.values()
# print(f'Optimal model has depth {depth}, n_estimators {n_estimators} and max_features {n_features}')

# # refit best model and show results
# rf_best = RandomForestClassifier(criterion='entropy',
#                                  max_features=n_features,
#                                  min_weight_fraction_leaf=0.05,
#                                  max_depth=depth,
#                                  n_estimators=n_estimators,
#                                  class_weight='balanced')
# rf_best.fit(X_train, y_train) # , sample_weight=labels_clean.reindex(X_train.index).t_value.abs()
# trademl.modeling.metrics_summary.display_clf_metrics(
#     rf_best, X_train, X_test, y_train, y_test)
# trademl.modeling.metrics_summary.plot_roc_curve(
#     rf_best, X_train, X_test, y_train, y_test)


# ### FEATURE SELECTION

# # copy fit object. Feature importnace function s can chane it.
# rf_best_ = sklearn.clone(rf_best)

# # mean decreasing impurity
# mdi_feature_imp = ml.feature_importance.mean_decrease_impurity(
#     rf_best_, X_train.columns)
# ml.feature_importance.plot_feature_importance(
#     mdi_feature_imp, 0, 0, save_fig=True,
#     output_path='mdi_feat_imp.png')

# # mean decreasing accuracy
# mda_feature_imp = ml.feature_importance.mean_decrease_accuracy(
#     rf_best_, X_train, y_train, cv_gen_purged,
#     scoring=log_loss,
#     sample_weight_train=labels_clean.reindex(X_train.index).t_value.abs())
# ml.feature_importance.plot_feature_importance(
#     mda_feature_imp, 0, 0, save_fig=True,
#     output_path='mda_feat_imp.png')

# # Shapley values
# # load JS visualization code to notebook
# shap.initjs()
# # explain the model's predictions using SHAP
# explainer = shap.TreeExplainer(model=rf_best, model_output='raw')
# shap_values = explainer.shap_values(X_train)
# # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_train.iloc[0,:])
# # summarize feature importance
# shap.summary_plot(shap_values, X_train, plot_type='bar')



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
# predictions_h2o = m.predict(test)
# predictions_h2o['predict'].table()
# m.confusion_matrix()  # train set
# print(m.model_performance(train))  # train set
# print(m.model_performance(test))  # test set

# # feature importance
# m = h2o.get_model(lb[0,"model_id"])
# feature_importance = m.varimp(use_pandas=True)
# try:
#     m.varimp_plot()
# except TypeError as te:
#     print(te)
# shap.initjs()
# contributions = m.predict_contributions(test)
# contributions_matrix = contributions.as_data_frame()
# shap_values_h2o = contributions_matrix.iloc[:,:-1]
# shap.summary_plot(shap_values, train.as_data_frame().drop(columns=['bin']), plot_type='bar')


# ### MULTICLASS TREND SCANNING LABELING

# # make c ++3 classes
# labels_3 = labels_clean.copy()
# idela_vector = np.repeat(1/3, 3)
# seq = np.linspace(labels_3.t_value.min(), labels_3.t_value.max(), 500)
# fixed_labels_diff = []
# for i in seq:
#     new_bins = np.where((labels_3.t_value > -(i/2)) & (labels_3.t_value < i), 0, labels_3.bin)
#     fixed_labels = pd.Series(new_bins).value_counts(normalize=True)
#     if len(fixed_labels) is not 3:
#         fixed_labels_diff.append(np.inf)
#         continue
#     fixed_labels_diff.append((fixed_labels - idela_vector).abs().sum())
# thresh_min = np.argmin(np.asarray(fixed_labels_diff))
# thresh_min = np.round(seq[thresh_min], 4)
# new_bins = np.where((labels_3.t_value > -(thresh_min/2)) & 
#                     (labels_3.t_value < thresh_min), 0, labels_3.bin)
# labels_3['bin'] = new_bins
# print(labels_3['bin'].value_counts())

# # prepare model
# X = spy.drop(columns=['close_orig']).reindex(labels_3.index)
# y = labels_3['bin']
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=0.10, 
#                                                     shuffle=False, 
#                                                     stratify=None)
# print(X_train.shape); print(y_train.shape)
# print(X_test.shape); print(y_test.shape)

# # model
# cv_gen_purged = ml.cross_validation.PurgedKFold(
#     n_splits=4,
#     samples_info_sets=labels_clean.reindex(X_train.index).t1)  # X_train.t1  labels.reindex(X_train).t1
# rf = RandomForestClassifier(criterion='entropy',
#                             min_weight_fraction_leaf=0.05,
#                             class_weight='balanced')
# clf = GridSearchCV(rf,
#                    param_grid=parameters,
#                    scoring='accuracy',
#                    n_jobs=12,
#                    cv=cv_gen_purged)
# clf.fit(X_train, y_train, sample_weight=labels_clean.reindex(X_train.index).t_value.abs())  # , sample_weight=labels_clean.reindex(X_train.index).t_value.abs()
# depth, n_features, n_estimators = clf.best_params_.values()
# print(f'Optimal model has depth {depth}, n_estimators {n_estimators} and max_features {n_features}')

# # refit best model and show results
# rf_best = RandomForestClassifier(criterion='entropy',
#                                  max_features=n_features,
#                                  min_weight_fraction_leaf=0.05,
#                                  max_depth=depth,
#                                  n_estimators=n_estimators,
#                                  class_weight='balanced')
# rf_best.fit(X_train, y_train, sample_weight=labels_clean.reindex(X_train.index).t_value.abs()) # , sample_weight=labels_clean.reindex(X_train.index).t_value.abs()
# sklearn.metrics.multilabel_confusion_matrix(y_test, rf_best.predict(X_test),
#                                             labels=['0', '1', '-1'])
# pd.Series(rf_best.predict(X_test)).value_counts()

# trademl.modeling.metrics_summary.display_clf_metrics(
#     rf_best, X_train, X_test, y_train, y_test, avg='micro')
# trademl.modeling.metrics_summary.plot_roc_curve(
#     rf_best, X_train, X_test, y_train, y_test)




# ### BACKTESTING

# # raw test set perfrormance
# time_range = pd.date_range(X_test.index[0], X_test.index[-1], freq='1Min')
# close = spy.close_orig.reindex(time_range).to_frame().dropna()
# close_cum = cumulative_returns(close)
# close_cum.plot()

# # predicitons and predictions_proba
# predictions = rf_best.predict(X_test)
# predictions = pd.Series(predictions, index=X_test.index)
# predictions.value_counts()
# predictions_proba = rf_best.predict_proba(X_test)
# predictions_proba_negative = pd.Series(predictions_proba[:, 0], index=X_test.index)

# # sell when model says to sel, buy otherwise
# positions = pd.concat([close, predictions.rename('position')], axis=1)
# positions.position.value_counts()

# @njit
# def enter_positions(positions):
#     for i in range(positions.shape[0]): # postitions.shape[0]
#         if i > 0:
#             if (positions[i-1, 1] == -1) or (positions[i, 1] == -1.0):
#                 positions[i, 1] = -1
#             else: 
#                 positions[i, 1] = 1
#         else:
#             positions[i, 1] = 1
#     return positions
            

# predictions = enter_positions(positions.values)
# predictions = pd.DataFrame(predictions, index=close.index, columns=['close', 'position'])
# predictions.position.value_counts()
# predictions['adjusted_close'] = np.where(predictions.position == 1, predictions.close, np.nan)
# predictions = predictions.fillna(method='ffill', axis=0)
# close_cum_strategy = cumulative_returns(predictions.adjusted_close)
# close_cum_strategy.plot()
# pd.concat([close_cum, close_cum_strategy], axis=1).plot()

# predictions[predictions.position == -1]

# close_cum_strategy[-1]
# close_cum.values[-1]


# # define arguments for backtest staticis functions (my and pyfolios)

# tml.modeling.backtest.minute_to_daily_return(
#     trbar_info, predictions, bet_sizes=None)

# # PYFOLIO
# pf.create_perf_attrib_tear_sheet(returns=daily_returns)
# pyfolio_statistics = pf.timeseries.perf_stats(
#     returns=daily_returns
# )