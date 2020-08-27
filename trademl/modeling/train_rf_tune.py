from pathlib import Path
from datetime import datetime
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import mlfinlab as ml
from mlfinlab.feature_importance import get_orthogonal_features
import trademl as tml
import mfiles
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform
from tensorboardX import SummaryWriter
from tune_sklearn import TuneSearchCV, TuneGridSearchCV



### TENSORBORADX WRITER
log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)


### MODEL HYPERPARAMETERS
input_data_path = 'D:/algo_trading_files'
rand_state = 3
sample_weights_type = 'return'
cv_type = 'purged_kfold'
cv_number = 5


### IMPORT PREPARED DATA
X_train = pd.read_pickle(os.path.join(Path(input_data_path), 'X_train.pkl'))
X_test = pd.read_pickle(os.path.join(Path(input_data_path), 'X_test.pkl'))
y_train = pd.read_pickle(os.path.join(Path(input_data_path), 'y_train.pkl'))
y_test = pd.read_pickle(os.path.join(Path(input_data_path), 'y_test.pkl'))
labeling_info = pd.read_pickle(os.path.join(Path(input_data_path), 'labeling_info.pkl'))


### SAMPLE WEIGHTS
if 't_value' in labeling_info.columns:
    sample_weights = labeling_info['t_value'].reindex(X_train.index).abs()
elif sample_weights_type == 'returns':
    sample_weights = ml.sample_weights.get_weights_by_return(
        labeling_info.reindex(X_train.index),
        X_train.loc[X_train.index, 'close_orig' if 'close_orig' in X_train.columns else 'close'],
        num_threads=1)
elif sample_weights_type == 'time_decay':
    sample_weights = ml.sample_weights.get_weights_by_time_decay(
        labeling_info.reindex(X_train.index),
        X_train.loc[X_train.index, 'close_orig' if 'close_orig' in X_train.columns else 'close'],
        decay=0.5, num_threads=1)
elif sample_weights_type == 'none':
    sample_weights = None


### CROS VALIDATION STEPS
if cv_type == 'purged_kfold':
    cv = ml.cross_validation.PurgedKFold(
        n_splits=cv_number,
        samples_info_sets=labeling_info['t1'].reindex(X_train.index))


### MODELING
# estimator
rf = RandomForestClassifier(criterion='entropy',
                            class_weight='balanced_subsample')

# grid search
# param_grid = {
#     'max_depth': [2, 3, 4, 5],
#     # 'n_estimators': [500, 1000],
#     # 'max_features': [5, 10, 15, 20],
#     # 'max_leaf_nodes': [4, 8, 16, 32]
#     }
# tune_search = TuneGridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     early_stopping=False,
#     scoring='accuracy',
#     n_jobs=12,
#     cv=cv,
#     verbose=1
# )
# tune_search.fit(X_train, y_train, sample_weight=sample_weights)

# random search
param_random = {
    "n_estimators": randint(50, 1000),
    "max_depth": randint(2, 7),
    'max_features': randint(5, 25),
    'min_weight_fraction_leaf': [0.0, 0.03,  0.05, 0.07, 0.10, 0.15],
    'min_impurity_decrease': [0.0, 0.00001,  0.0001, 0.001, 0.01, 0.1]
}
tune_search = TuneSearchCV(
    estimator=rf,
    param_distributions=param_random,
    search_optimization="random",
    early_stopping=False,
    n_iter=30,
    scoring='accuracy',
    n_jobs=12,
    cv=cv,
    verbose=1
)
tune_search.fit(X_train, y_train, sample_weight=sample_weights)

# bayesian search
tune_search = TuneSearchCV(
    rf,
    param_random,
    search_optimization='bayesian',
    max_iters=100,
    scoring='accuracy',
    n_jobs=12,
    cv=cv,
    verbose=1
)
tune_search.fit(X_train, y_train, sample_weight=sample_weights)

# scores
clf_predictions = tune_search.predict(X_test)
tune_search.best_params_
print(tune_search.cv_results_)
tune_search.cv_results_['mean_test_score'].mean()
tune_search.best_score_


# model scores
clf_f1_score = sklearn.metrics.f1_score(y_test, clf_predictions)
clf_accuracy_score = sklearn.metrics.accuracy_score(y_test, clf_predictions)
print(f'f1_score: {clf_f1_score}')
# print(f'optimal_max_depth: {max_depth}')
# print(f'optimal_n_features: {n_features}')
# print(f'optimal_max_leaf_nodes {max_leaf_nodes}')
# print(f'optimal_n_estimators {n_estimators}')
# save_id = f'{max_depth}{n_features}{max_leaf_nodes}{n_estimators}{str(clf_f1_score)[2:6]}'
