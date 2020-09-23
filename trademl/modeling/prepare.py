from pathlib import Path
import os
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import mlfinlab as ml
from mlfinlab.feature_importance import get_orthogonal_features
import trademl as tml
import mfiles
from tensorboardX import SummaryWriter
from datetime import datetime
from pycaret.preprocess import Zroe_NearZero_Variance, Fix_multicollinearity
from trademl.modeling.structural_breaks import ChowStructuralBreakSubsample
from trademl.modeling.stationarity import StationarityMethod
from scipy.signal import savgol_filter
matplotlib.use("Agg")  # don't show graphs because thaty would stop guildai script
print('Start prepare step')


### TENSORBORADX WRITER
log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

### HYPERPARAMETERS
# load and save data
contract = 'SPY_IB'
input_data_path = 'D:/market_data/usa/ohlcv_features'
env_directory = None # os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
# subsample
chow_subsample = None
stationarity_tecnique = 'fracdiff'
# structural breaks
structural_break_regime = 'all'
# labeling
label_tuning = True
label = 'day_30'  # 'day_1' 'day_2' 'day_5' 'day_10' 'day_20' 'day_30' 'day_60'
labeling_technique = 'tb'  # tb is triple-barrier; ts is trend scanning
tb_triplebar_num_days = 4
tb_triplebar_pt_sl = [1, 1]
tb_triplebar_min_ret = 0.005
ts_look_forward_window = 1200  # 60 * 8 * 10 (10 days)
ts_min_sample_length = 30
ts_step = 5
tb_min_pct = 0.05
# filtering
tb_volatility_lookback = 50
tb_volatility_scaler = 1
train_test_split_ratio = 0.1
time_step_length = 10
# feature engineering
correlation_threshold = 0.99    
pca = True
# scaling
scaling = 'none'
# performance
num_threads = 1


# import data
file_name = contract + '_clean'
data = pd.read_hdf(os.path.join(Path(input_data_path), file_name + '.h5'), file_name)
data.sort_index(inplace=True)

# Choose subsamples, stationarity method and make labels
pipe = make_pipeline(
    ChowStructuralBreakSubsample(min_length=10) if chow_subsample else None,
    StationarityMethod(stationarity_method='fracdiff'),
    )
data = pipe.fit_transform(data)

# categorical variables
categorial_features = ['tick_rule', 'HT_TRENDMODE', 'volume_vix']
categorial_features = [col for col in categorial_features if col in data.columns]
data = data.drop(columns=categorial_features)  # remove for now

# Labeling
if label_tuning:
    if labeling_technique == 'tb':
        triple_barrier_pipe= tml.modeling.pipelines.TripleBarierLabeling(
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
    elif labeling_technique == 'ts':
        trend_scanning_pipe = tml.modeling.pipelines.TrendScanning(
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
        labeling_info = ml.labeling.fixed_time_horizon(data['orig_close'], threshold=0.005, resample_by='B').dropna().to_frame()
        labeling_info = labeling_info.rename(columns={'orig_close': 'bin'})
        print(labeling_info.iloc[:, 0].value_counts())
        X = X.iloc[:-1, :]
else:
    X_cols = [col for col in data.columns if 'day_' not in col]
    X = data[X_cols]
    y_cols = [col for col in data.columns if label + '_' in col]
    labeling_info = data[y_cols]
    # filtering
    daily_vol = ml.util.get_daily_vol(data['orig_close' if 'orig_close' in data.columns else 'close'], lookback=50)
    cusum_events = ml.filters.cusum_filter(data['orig_close' if 'orig_close' in data.columns else 'close'], threshold=daily_vol.mean()*1)
    X = X.reindex(cusum_events)
    labeling_info = labeling_info.reindex(cusum_events)

# remove na
remove_na_rows = labeling_info.isna().any(axis=1)
X = X.loc[~remove_na_rows]
labeling_info = labeling_info.loc[~remove_na_rows]
labeling_info.iloc[:, -1] = np.where(labeling_info.iloc[:, -1] == -1, 0, labeling_info.iloc[:, -1])

# Removing large values (TA issue - causes model problems / overflow)
# X.apply(lambda x: any((x >= 1e12) | (x <= -1e12)), axis=0)


### REMOVE CORRELATED ASSETS
X = tml.modeling.preprocessing.remove_correlated_columns(
    data=X,
    columns_ignore=[],
    threshold=correlation_threshold)


### TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, labeling_info.loc[:, labeling_info.columns.str.contains('bin')],
    test_size=train_test_split_ratio, shuffle=False, stratify=None)


### SCALING
if scaling == 'expanding':
    stdize_input = lambda x: (x - x.expanding(tb_volatility_lookback).mean()) / x.expanding(tb_volatility_lookback).std()
    X_train = X_train.apply(stdize_input)
    X_test = X_test.apply(stdize_input)
    y_train = y_train.loc[~X_train.isna().any(axis=1)]
    X_train = X_train.dropna()
    y_test = y_test.loc[~X_test.isna().any(axis=1)]
    X_test = X_test.dropna()

    y_train = y_train.loc[~X_train.isna().any(axis=1)]
    X_train = X_train.dropna()
    y_test = y_test.loc[~X_test.isna().any(axis=1)]
    X_test = X_test.dropna()



# from gplearn.genetic import SymbolicTransformer
# from gplearn.functions import make_function


# def exponent(x):
#     with np.errstate(over='ignore'):
#         return np.where(np.abs(x) < 100, np.exp(x), 0.)


# class Genetic(BaseEstimator, TransformerMixin):

#     def __init__(self, population=50000, generations=10, hall_of_fame=500, components=200, metric='spearman'):
#         self.state = {}
#         self.population = population
#         self.generations = generations
#         self.hall_of_fame = hall_of_fame
#         self.components = components
#         self.metric = metric

#         # population: Number of formulas per generation
#         # generations: Number of generations
#         # hall_of_fame: Best final evolution program to evaluate
#         # components: X least correlated from the hall of fame
#         # metric: pearson for linear model, spearman for tree based estimators

#     def fit(self, X, y=None, state={}):
#         exponential = make_function(function=exponent, name='exp', arity=1)

#         function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max',
#                         'min', 'tan', 'sin', 'cos', exponential]

#         gp = SymbolicTransformer(generations=self.generations, population_size=self.population,
#                                  hall_of_fame=self.hall_of_fame, n_components=self.components,
#                                  function_set=function_set,
#                                  parsimony_coefficient='auto',
#                                  max_samples=0.6, verbose=1, metric=self.metric,
#                                  random_state=0, n_jobs=7)

#         self.state['genetic'] = {}
#         self.state['genetic']['fit'] = gp.fit(X, y)

#         return self

#     def transform(self, X, y=None, state={}):
#         features = self.state['genetic']['fit'].transform(X)
#         features = pd.DataFrame(features, columns=["genetic_" + str(a) for a in range(features.shape[1])], index=X.index)
#         X = X.join(features)

#         return X, y, self.state


# X_train_sample = X_train[:1000]
# y_train_sample = y_train[:1000]
# gen = Genetic()
# test_gen = gen.fit_transform(X_train_sample, y_train_sample)
# X_train.shape
# test_gen[0].shape


### DIMENSIONALITY REDUCTION
if pca:
    if scaling == 'none':
        X_train = pd.DataFrame(preprocessing.scale(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(preprocessing.scale(X_test), columns=X_test.columns)
    X_train = pd.DataFrame(
        get_orthogonal_features(X_train),
        index=X_train.index).add_prefix("PCA_")
    pca_n_compenents = X_train.shape[1]
    X_test = pd.DataFrame(
        get_orthogonal_features(X_test,
                                num_features=pca_n_compenents),
        index=X_test.index).add_prefix("PCA_")
    X_train.index = y_train.index
    X_test.index = y_test.index


### ADD close if it does not exists, needed for later
if 'close' not in X_train.columns:
    X_train = X_train.join(X['close'], how='left')
    X_test = X_test.join(X['close'], how='left')


### SAVE FILES
# save localy   
file_names = ['X_train', 'y_train', 'X_test', 'y_test', 'labeling_info']
saved_files = [X_train, y_train, X_test, y_test, labeling_info]
file_names_pkl = [f + '.pkl' for f in file_names]
X_train.to_pickle('X_train.pkl')
tml.modeling.utils.save_files(
    saved_files,
    file_names_pkl,
    Path('./'))
X_train.to_pickle('X_train.pkl')
# save to mfiles
if env_directory is not None:
    mfiles_client = tml.modeling.utils.set_mfiles_client(env_directory)
    tml.modeling.utils.destroy_mfiles_object(mfiles_client, file_names)
    for f in file_names:
        mfiles_client.upload_file(file_names_pkl, object_type='Dokument')
print('End prepare step')
