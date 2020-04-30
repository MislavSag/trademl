# fundamental modules
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
from IPython.display import display, Image
# preprocessing
from sklearn.model_selection import train_test_split
# modelling
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



DATA_PATH = 'C:/Users/Mislav/algoAItrader/data/spy_with_vix.h5'

### IMPORT DATA
with pd.HDFStore(DATA_PATH) as store:
    spy = store.get('spy_with_vix')
spy.sort_index(inplace=True)
spy = spy.iloc[:30000]

### CHOOSE/REMOVE VARIABLES
remove_ohl = ['open', 'low', 'high', 'vixFirst', 'vixHigh', 'vixLow']  # correlatin > 0.99
spy.drop(columns=remove_ohl, inplace=True)  #correlated with close


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


### REMOVE OUTLIERS
outlier_remove = tml.modeling.pipelines.OutlierStdRemove(std_outlier)
spy = outlier_remove.fit_transform(spy)


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
tb_fit = triple_barrier_pipe.fit(spy)
X = tb_fit.transform(spy)


### TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X.drop(columns=['close_orig']), tb_fit.triple_barrier_info['bin'],
    test_size=0.10, shuffle=False, stratify=None)


### SAMPLE WEIGHTS (DECAY FACTOR CAN BE ADDED!)
if sample_weights_type == 'returns':
    sample_weigths = ml.sample_weights.get_weights_by_return(
        tb_fit.triple_barrier_info.reindex(X_train.index),
        spy.loc[X_train.index, 'close_orig'],
        num_threads=1)
elif sample_weights_type == 'time_decay':
    sample_weigths = ml.sample_weights.get_weights_by_time_decay(
        tb_fit.triple_barrier_info.reindex(X_train.index),
        spy.loc[X_train.index, 'close_orig'],
        decay=0.5, num_threads=1)


### CROS VALIDATION STEPS
if cv_type == 'purged_kfold':
    cv = ml.cross_validation.PurgedKFold(
        n_splits=cv_number,
        samples_info_sets=tb_fit.triple_barrier_info.reindex(X_train.index).t1)


# MODEL
rf_best = RandomForestClassifier(criterion='entropy',
                                 max_features=max_features,
                                 min_weight_fraction_leaf=0.05,
                                 max_depth=max_depth,
                                 n_estimators=n_estimators,
                                 class_weight='balanced_subsample',
                                 random_state=rand_state)
rf_best.fit(X_train, y_train, sample_weight=sample_weigths)


# def train():
#     data = mnist_data.load()
#     model = mnist_model.init()
#     model.train(data)

# if __name__ == "__main__":
#     train()