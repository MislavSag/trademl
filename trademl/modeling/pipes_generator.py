import pandas as pd
import trademl as tml
from trademl.modeling.data_import import import_ohlcv

def import_ohlcv(path, contract='SPY_IB'):
    cache_path = os.path.join(Path(path), 'cache', contract + '.h5')
    if os.path.exists(cache_path):
        security = pd.read_hdf(cache_path, contract)
        q = 'SELECT date, open, high, low, close, volume, average, barCount FROM ' + contract + ' ORDER BY id DESC LIMIT 1'
        data = tml.modeling.utils.query_to_db(q, 'odvjet12_market_data_usa')
        if not (data['date'] == security.index[-1])[0]:        
            q = 'SELECT date, open, high, low, close, volume, average, barCount FROM ' + contract
            data = tml.modeling.utils.query_to_db(q, 'odvjet12_market_data_usa')
            data.set_index(data.date, inplace=True)
            data.drop(columns=['date'], inplace=True)
            security = data.sort_index()
            cache_path = os.path.join(Path(path), 'cache', contract + '.h5')
            security.to_hdf(cache_path, contract)
     
    return data
    

data = tml.modeling.data_import.import_ohlcv(
    'D:/market_data/usa/ohlcv_features', 
    contract='SPY_IB')

data = import_ohlcv(path='D:/market_data/usa/ohlcv_features', contract='SPY_IB')

### IMPORT DATA
# # import data from mysql database and 


# ################# MOVE LATER TO OTHER FOLDER #################

# ### IMPORT DATA
# def import_data(data_path, remove_cols, contract='SPY'):
#     # import data
#     with pd.HDFStore(os.path.join(data_path, contract + '.h5')) as store:
#         data = store.get(contract)
#     data.sort_index(inplace=True)
    
#     # remove variables
#     remove_cols = [col for col in remove_cols if col in data.columns]
#     data.drop(columns=remove_cols, inplace=True)
    
#     return data





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
