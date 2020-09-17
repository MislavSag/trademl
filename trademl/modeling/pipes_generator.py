import pandas as pd
import trademl as tml
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from trademl.modeling.data_import import import_ohlcv
from trademl.modeling.outliers import RemoveOutlierDiffMedian

# Import data
data = import_ohlcv('D:/market_data/usa/ohlcv_features', contract='SPY_IB')

# Split data
X_train, X_test = train_test_split(
    data, test_size=0.10, shuffle=False, stratify=None)

# Preprocessing
pipe = make_pipeline(
    RemoveOutlierDiffMedian(median_outlier_thrteshold=25),
    AddFeatures(ta_periods=[10, 20])
    )
X = pipe.fit_transform(X_train)


# class AddFeatures(BaseEstimator, TransformerMixin):

#     def __init__(self, add_ta=True, ta_periods=[10, 100]):
#         self.add_ta = add_ta
#         self.ta_periods = ta_periods

#     def fit(self, X, y=None):

#         return self

#     def transform(self, X, y=None):
#         # add tecnical indicators        
#         if self.add_ta:
#             X = tml.modeling.features.add_technical_indicators(X, periods=self.ta_periods)
#             X.columns = [cl[0] if isinstance(cl, tuple) else cl for cl in X.columns]
        
#         # add other features
#         X = tml.modeling.features.add_ohlcv_features(X)
        
#         # remove na
#         if self.add_ta:
#             X = X.loc[:, X.isna().sum() < (max(self.ta_periods) + 10)]
#         cols_remove_na = range((np.where(X.columns == 'volume')[0].item() + 1), X.shape[1])
#         X.dropna(subset=X.columns[cols_remove_na], inplace=True)
        
#         return X


af = AddFeatures(ta_periods=[10, 20])
X = af.fit_transform(X)



tb1 = TripleBarrierLabel()
tb1_fit = tb1.fit(X)
tb1_transform = tb1.transform(X)
tb1_transform

tb2 = tml.modeling.pipelines.TripleBarierLabeling()
tb2_fit = tb2.fit(X)
tb1_transform = tb1.transform(X)



### 1) REMOVE OUTLIERS
# security = tml.modeling.outliers.remove_ourlier_diff_median(data, median_outlier_thrteshold)



# from sklearn.base import BaseEstimator, TransformerMixin
# from trademl.modeling.outliers import remove_ourlier_diff_median


# class RemoveOutlierDiffMedian(BaseEstimator, TransformerMixin):

#     def __init__(self, median_outlier_thrteshold, state={}):
#         self.median_outlier_thrteshold = median_outlier_thrteshold
#         self.state = state

#     def fit(self, X, y=None, state={}):
#         if type(X) is tuple: X, y, self.state = X
#         return self

#     def transform(self, X, y=None, state={}):
#         if type(X) is tuple: X, y, self.state = X

#         X = remove_ourlier_diff_median(X, self.median_outlier_thrteshold)

#         return X


data_sample = data.iloc[:10000]
remove_ourlier = RemoveOutlierDiffMedian(median_outlier_thrteshold=25)
data_sample = remove_ourlier.fit_transform(data_sample)
data_sample.head()


# class ChowStructuralBreakSubsample(BaseEstimator, TransformerMixin):

#     def __init__(self, min_length=10, state={}):
#         self.min_length = min_length
#         self.state = state

#     @time_method
#     def fit(self, X, y=None, state={}):
#         if type(X) is tuple: X, y, self.state = X
        
#         # extract close series
#         assert 'close' in X.columns, "Dataframe doesn't contain close column"
#         close = X['close']
        
#         # convert to weekly freq for performance and noise reduction
#         close_weekly = security['close'].resample('W').last().dropna()
#         close_weekly_log = np.log(close_weekly)
        
#         # calculate chow indicator
#         chow = tml.modeling.structural_breaks.get_chow_type_stat(
#             series=close_weekly_log, min_length=self.min_length)
        
#         # take second segment of structural break
#         breakdate = chow.loc[chow == chow.max()]
#         X['chow_segment'] = 0
#         X['chow_segment'][breakdate.index[0]:] = 1
#         X['chow_segment'].loc[breakdate.index[0]:] = 1
#         X['chow_segment'] = np.where(X.index < breakdate.index[0], 0, 1)

#         return self

#     @time_method
#     def transform(self, X, y=None, state={}):
#         if type(X) is tuple: X, y, self.state = X
        
#         # subsample
#         if structural_break_regime == 'chow':
#             if (X.loc[X['chow_segment'] == 1].shape[0] / 60 / 8) < 365:
#                 X = X.iloc[-(60*8*365):]
#             else:
#                 data = data.loc[data['chow_segment'] == 1]
#         data = data.drop(columns=['chow_segment'])
        
#         return X, y, self.state


# data_sample = data.iloc[:10000]
# data_sample = data_sample.rename(columns={'orig_close': 'close'})

# chow_pipe = ChowStructuralBreakSubsample(
#     min_length=10
# )
# tb_fit = chow_pipe.fit(data_sample)
# X = tb_fit.transform(data_sample)

# ################# MOVE LATER TO OTHER FOLDER #################

# # Import data
# contract = 'SPY_raw_ta' if include_ta else 'SPY_raw'
# contract = 'SPY_raw_ta' if label_tuning else contract + '_labels'
# data = import_data(input_data_path, [], contract=contract)
# # data = tml.modeling.utils.import_data(input_data_path, [], contract=contract)


# ### REGIME DEPENDENT ANALYSIS
# if structural_break_regime == 'chow':
#     if (data.loc[data['chow_segment'] == 1].shape[0] / 60 / 8) < 365:
#         data = data.iloc[-(60*8*365):]
#     else:
#         data = data.loc[data['chow_segment'] == 1]
# data = data.drop(columns=['chow_segment'])


# ### CHOOSE STATIONARY / UNSTATIONARY
# if stationarity_tecnique == 'fracdiff':
#     remove_cols = [col for col in data.columns if 'orig_' in col and col != 'orig_close']  
# elif stationarity_tecnique == 'orig':
#     remove_cols = [col for col in data.columns if 'fracdiff_' in col and col != 'orig_close']
# data = data.drop(columns=remove_cols)


# ### LABELLING
# if label_tuning:
#     if labeling_technique == 'triple_barrier':
#         # TRIPLE BARRIER LABELING
#         triple_barrier_pipe= tml.modeling.pipelines.TripleBarierLabeling(
#             close_name='orig_close' if 'orig_close' in data.columns else 'close',
#             volatility_lookback=tb_volatility_lookback,
#             volatility_scaler=tb_volatility_scaler,
#             triplebar_num_days=tb_triplebar_num_days,
#             triplebar_pt_sl=tb_triplebar_pt_sl,
#             triplebar_min_ret=tb_triplebar_min_ret,
#             num_threads=num_threads,
#             tb_min_pct=tb_min_pct
#         )   
#         tb_fit = triple_barrier_pipe.fit(data)
#         labeling_info = tb_fit.triple_barrier_info
#         X = tb_fit.transform(data)
#     elif labeling_technique == 'trend_scanning':
#         trend_scanning_pipe = tml.modeling.pipelines.TrendScanning(
#             close_name='orig_close' if 'orig_close' in data.columns else 'close',
#             volatility_lookback=tb_volatility_lookback,
#             volatility_scaler=tb_volatility_scaler,
#             ts_look_forward_window=ts_look_forward_window,
#             ts_min_sample_length=ts_min_sample_length,
#             ts_step=ts_step
#             )
#         labeling_info = trend_scanning_pipe.fit(data)
#         X = trend_scanning_pipe.transform(data)
#     elif labeling_technique == 'fixed_horizon':
#         X = data.copy()
#         labeling_info = ml.labeling.fixed_time_horizon(data['orig_close'], threshold=0.005, resample_by='B').dropna().to_frame()
#         labeling_info = labeling_info.rename(columns={'orig_close': 'bin'})
#         print(labeling_info.iloc[:, 0].value_counts())
#         X = X.iloc[:-1, :]
# else:
#     X_cols = [col for col in data.columns if 'day_' not in col]
#     X = data[X_cols]
#     y_cols = [col for col in data.columns if label + '_' in col]
#     labeling_info = data[y_cols]


# ### FILTERING
# if label_tuning:
#     X = X.drop(columns=['orig_close'])
# else:
#     daily_vol = ml.util.get_daily_vol(data['orig_close' if 'orig_close' in data.columns else 'close'], lookback=50)
#     cusum_events = ml.filters.cusum_filter(data['orig_close' if 'orig_close' in data.columns else 'close'], threshold=daily_vol.mean()*1)
#     X = X.reindex(cusum_events)
#     labeling_info = labeling_info.reindex(cusum_events)
# ### ZAVRSITI DO KRAJA ####


# ### REMOVE NA
# remove_na_rows = labeling_info.isna().any(axis=1)
# X = X.loc[~remove_na_rows]
# labeling_info = labeling_info.loc[~remove_na_rows]
# labeling_info.iloc[:, -1] = np.where(labeling_info.iloc[:, -1] == -1, 0, labeling_info.iloc[:, -1])
# # labeling_info.iloc[:, -1] = labeling_info.iloc[:, -1].astype(pd.Int64Dtype())


# ### REMOVE CORRELATED ASSETS
# msg = f'Shape before removing correlated features with threshold {correlation_threshold}' \
#       f' is {X.shape} and after is'
# print(msg)
# X = tml.modeling.preprocessing.remove_correlated_columns(
#     data=X,
#     columns_ignore=[],
#     threshold=correlation_threshold)
# print(X.shape)


# # TREAT CATEGORIAL VARIABLES
# categorial_features = ['tick_rule', 'HT_TRENDMODE', 'volume_vix']
# categorial_features = [col for col in categorial_features if col in X.columns]
# X = X.drop(columns=categorial_features)  # remove for now


# ### TRAIN TEST SPLIT
# X_train, X_test, y_train, y_test = train_test_split(
#     X, labeling_info.loc[:, labeling_info.columns.str.contains('bin')],
#     test_size=0.10, shuffle=False, stratify=None)


# ### SCALING
# if scaling == 'expanding':
#     stdize_input = lambda x: (x - x.expanding(tb_volatility_lookback).mean()) / x.expanding(tb_volatility_lookback).std()
#     X_train = X_train.apply(stdize_input)
#     X_test = X_test.apply(stdize_input)
#     y_train = y_train.loc[~X_train.isna().any(axis=1)]
#     X_train = X_train.dropna()
#     y_test = y_test.loc[~X_test.isna().any(axis=1)]
#     X_test = X_test.dropna()

#     y_train = y_train.loc[~X_train.isna().any(axis=1)]
#     X_train = X_train.dropna()
#     y_test = y_test.loc[~X_test.isna().any(axis=1)]
#     X_test = X_test.dropna()

    
# ### DIMENSIONALITY REDUCTION
# if pca:
#     if scaling == 'none':
#         X_train = pd.DataFrame(preprocessing.scale(X_train), columns=X_train.columns)
#         X_test = pd.DataFrame(preprocessing.scale(X_test), columns=X_test.columns)
#     X_train = pd.DataFrame(
#         get_orthogonal_features(X_train),
#         index=X_train.index).add_prefix("PCA_")
#     pca_n_compenents = X_train.shape[1]
#     X_test = pd.DataFrame(
#         get_orthogonal_features(X_test,
#                                 num_features=pca_n_compenents),
#         index=X_test.index).add_prefix("PCA_")
#     X_train.index = y_train.index
#     X_test.index = y_test.index


# ### SAMPLE WEIGHTS
# labeling_info.columns = labeling_info.columns.str.replace(r'day_\d+_', '', regex=True)
# if 't_value' in labeling_info.columns:
#     sample_weights = labeling_info['t_value'].reindex(X_train.index).abs()
# elif sample_weights_type == 'returns':
#     sample_weights = ml.sample_weights.get_weights_by_return(
#         labeling_info.reindex(X_train.index),
#         X_train.loc[X_train.index, 'close_orig' if 'close_orig' in X_train.columns else 'close'],
#         num_threads=1)
# elif sample_weights_type == 'time_decay':
#     sample_weights = ml.sample_weights.get_weights_by_time_decay(
#         labeling_info.reindex(X_train.index),
#         X_train.loc[X_train.index, 'close_orig' if 'close_orig' in X_train.columns else 'close'],
#         decay=0.5, num_threads=1)
# elif sample_weights_type == 'none':
#     sample_weights = None


# ### CROS VALIDATION STEPS
# if cv_type == 'purged_kfold':
#     cv = ml.cross_validation.PurgedKFold(
#         n_splits=cv_number,
#         samples_info_sets=labeling_info['t1'].reindex(X_train.index))

# ### MODEL
# clf = RandomForestClassifier(criterion='entropy',
#                                 # max_features=max_features,
#                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
#                                 max_depth=max_depth,
#                                 n_estimators=n_estimators,
#                                 class_weight=class_weight,
#                                 # random_state=rand_state,
#                                 n_jobs=16)
# scores = ml.cross_validation.ml_cross_val_score(
#     clf, X_train, y_train, cv_gen=cv, 
#     sample_weight_train=sample_weights,
#     scoring=sklearn.metrics.accuracy_score)  #sklearn.metrics.f1_score(average='weighted')


# ### CV RESULTS
# mean_score = scores.mean()
# std_score = scores.std()
# save_id = str(random.getrandbits(32))
# # save_id = f'{max_depth}{max_features}{n_estimators}{str(mean_score)[2:6]}'
# print(f'Mean score: {mean_score}')
# writer.add_scalar(tag='mean_score', scalar_value=mean_score, global_step=None)
# writer.add_scalar(tag='std_score', scalar_value=std_score, global_step=None)
# writer.add_text(tag='save_id', text_string=save_id, global_step=None)

# # retrain the model if mean score is high enough (higher than 0.5)
# if mean_score < 0.55:
#     print('good_performance: False')
# else:
#     print('good_performance: True')
    
#     # refit the model and get results
#     clf = RandomForestClassifier(criterion='entropy',
#                                 max_features=max_features,
#                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
#                                 max_depth=max_depth,
#                                 n_estimators=n_estimators,
#                                 class_weight=class_weight,
#                                 # random_state=rand_state,
#                                 n_jobs=16)
#     clf.fit(X_train, y_train, sample_weight=sample_weights)
#     tml.modeling.metrics_summary.clf_metrics_tensorboard(
#         writer, clf, X_train, X_test, y_train, y_test, avg='binary')

#     # save feature importance tables and plots
#     shap_values, importances, mdi_feature_imp = tml.modeling.feature_importance.important_features(
#         clf, X_train, y_train, save_id,
#         save_path=os.path.join(Path(input_data_path), 'fi_plots'))
#     tml.modeling.utils.save_files([shap_values, importances, mdi_feature_imp],
#                 file_names=[f'shap_{save_id}.csv',
#                             f'rf_importance_{save_id}.csv',
#                             f'mpi_{save_id}.csv'],
#                 directory=os.path.join(Path(input_data_path), 'important_features'))
    
    
#     ### REFIT THE MODEL WITH MOST IMPORTANT FEATURES
#     fi_cols = shap_values['col_name'].head(keep_important_features)
#     X_train_important = X_train[fi_cols]
#     X_test_important = X_test[fi_cols]
#     clf = RandomForestClassifier(criterion='entropy',
#                             max_features=keep_important_features,
#                             min_weight_fraction_leaf=min_weight_fraction_leaf,
#                             max_depth=max_depth,
#                             n_estimators=n_estimators,
#                             class_weight=class_weight,
#                             # random_state=rand_state,
#                             n_jobs=16)
#     clf_important = clf.fit(X_train_important, y_train, sample_weight=sample_weights)
#     tml.modeling.metrics_summary.clf_metrics_tensorboard(
#         writer, clf_important, X_train_important,
#         X_test_important, y_train, y_test, avg='binary', prefix='fi_')

# # close writer
# writer.close()
