from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import shap
import lightgbm as lgb
import mlfinlab as ml
import trademl as tml
from tensorboardX import SummaryWriter
import random
import re
matplotlib.use("Agg")  # don't show graphs


### TENSORBORADX WRITER
log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)


### MODEL HYPERPARAMETERS
sample_weights_type = 'returns'
cv_type = 'purged_kfold'
cv_number = 5
# model
boosting_type = 'gbdt'
num_leaves  = 50
n_estimators  = 500
min_child_samples = 10
subsample = 0.95
max_depth = 3
learning_rate = 0.1
colsample_bytree = 0.9
bagging_fraction = 1.0
lambda_l1 = 0.1


### IMPORT PREPARED DATA
X_train = pd.read_pickle('X_train.pkl')
X_test = pd.read_pickle('X_test.pkl')
y_train = pd.read_pickle('y_train.pkl')
y_test = pd.read_pickle('y_test.pkl')
labeling_info = pd.read_pickle('labeling_info.pkl')


### SAMPLE WEIGHTS
labeling_info.columns = labeling_info.columns.str.replace(r'day_\d+_', '', regex=True)
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

# Remove close column if pcais used
if re.search('PCA', X_train.iloc[:,[0]].columns.values.item()):
    X_train = X_train.drop(columns=['close'])
    X_test = X_test.drop(columns=['close'])

### CROS VALIDATION STEPS
if cv_type == 'purged_kfold':
    cv = ml.cross_validation.PurgedKFold(
        n_splits=cv_number,
        samples_info_sets=labeling_info['t1'].reindex(X_train.index))


### MODEL
# convert pandas df to lbt matrix
train_set = lgb.Dataset(data=X_train, label=y_train.astype(int))
test_set = lgb.Dataset(data=X_test, label=y_test.astype(int))

# parameters for GridSearch
params = {
    'boosting_type': boosting_type,
    'num_leaves': num_leaves,
    'max_depth': max_depth,
    'subsample': subsample,
    'max_depth': max_depth,
    'learning_rate': learning_rate,
    'n_estimators': n_estimators,
    'min_child_samples': min_child_samples,
    'subsample': subsample,
    'colsample_bytree': colsample_bytree,
    'bagging_fraction': bagging_fraction,
    'lambda_l1': lambda_l1
    }

# cv
cv_clf = lgb.cv(
    params=params,
    train_set=train_set,
    num_boost_round = 500,  # Here we will use a large number again and count on early_stopping_rounds to find the optimal number of rounds before reaching the maximum.
    folds=cv,
    early_stopping_rounds=15,
    metrics="auc"
    )

# cv scores
cv_results = pd.DataFrame(cv_clf)
cv_results = cv_results.loc[cv_results.iloc[:, 0] == cv_results.iloc[:, 0].max()]
mean_score = cv_results.iloc[0, 0]
std_score = cv_results.iloc[0, 1]
save_id = str(random.getrandbits(32))
print(f'Mean score: {mean_score}')
writer.add_scalar(tag='mean_score', scalar_value=mean_score, global_step=None)
writer.add_scalar(tag='std_score', scalar_value=std_score, global_step=None)
writer.add_text(tag='save_id', text_string=save_id, global_step=None)


# Continue i=only if score is high enough
if mean_score > 0.55:
    
    # split X_train data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train.astype(int),
        test_size=0.15, shuffle=False, stratify=None)
    sample_weights_train=sample_weights.iloc[:X_train.shape[0]]

    # classifier
    clf = lgb.LGBMClassifier(
        boosting_type=boosting_type,
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        objective='binary',
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        bagging_fraction=bagging_fraction,
        lambda_l1=lambda_l1
        )

    # clf fit
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight=sample_weights_train,
            eval_metric='auc', early_stopping_rounds=30, verbose=True)

    # eval scores
    evals_result = clf.evals_result_['valid_0']
    evals_result = pd.DataFrame(evals_result)
    evals_result = evals_result.iloc[:, 1]
    best_score = evals_result.max()
    print(f'Best score: {best_score}')
    writer.add_scalar(tag='best_score', scalar_value=best_score, global_step=None)

    # Continue i=only if score is high enough
    if best_score > 0.55:
        # test scores
        clf_predictions = clf.predict(X_test)
        tml.modeling.metrics_summary.clf_metrics_tensorboard(
            writer, clf, X_train, X_test, y_train, y_test, avg='binary')

        # save important featues
        tml.modeling.feature_importance.fi_shap(clf, X_train, y_train, save_id, './')
        # fi_shap(clf, X_train, y_train, save_id, input_data_path)
        tml.modeling.feature_importance.fi_lightgbm(clf, X_train, save_id, './')
        # fi_lightgbm(clf, X_train, save_id, input_data_path)


