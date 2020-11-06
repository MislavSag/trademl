# fundamental modules
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import shap
import mlfinlab as ml
from mlfinlab.feature_importance import get_orthogonal_features
import trademl as tml
from tensorboardX import SummaryWriter
import random
import re
from BorutaShap import BorutaShap
import tscv
from sklearn.model_selection import cross_val_score
from rfpimp import *
matplotlib.use("Agg")  # don't show graphs


### TENSORBORADX WRITER
log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)


### MODEL HYPERPARAMETERS
sample_weights_type = 'returns'
cv_type = 'purged_kfold'  # GapWalkForward
cv_number = 5
max_depth = 2
max_features = 10
n_estimators = 350
min_weight_fraction_leaf = 0.1  
class_weight = 'balanced_subsample'
keep_important_features = 20  # for feature selection


# Import data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
y_val = np.load('y_val.npy')
col_names = pd.read_csv('col_names.csv')
col_names = col_names.iloc[:, 1]
Y = pd.read_pickle('Y.pkl')


# Convert to 2d WORKS WITH ONE FEATURE, NEED WORK TO WORK WITH MULTIPLIE FEATURES
X_train = X_train.reshape(-1, 5)
y_train = y_train.squeeze()
X_val = X_val.reshape(-1, 5)
y_val = y_val.squeeze()
X_test = X_test.reshape(-1, 5)
y_test = y_test.squeeze()
X_train = np.vstack((X_train, X_val))
y_train = np.concatenate((y_train, y_val))

# convert to pandas dataframe
pd_names = [col + str(i-4) for col in col_names for i in range(5)]
X_train = pd.DataFrame(X_train, columns=pd_names)
X_test = pd.DataFrame(X_test, columns=pd_names)


# Sample weigths
if not Y['ret'].isna().all():
    Y.columns = Y.columns.str.replace(r'day_\d+_', '', regex=True)
    if 't_value' in Y.columns:
        sample_weights = Y['t_value'].reindex(X_train.index).abs()
    elif sample_weights_type == 'returns':
        sample_weights = ml.sample_weights.get_weights_by_return(
            Y.reindex(X_train.index),
            X_train.loc['close'],
            num_threads=1)
    elif sample_weights_type == 'time_decay':
        sample_weights = ml.sample_weights.get_weights_by_time_decay(
            Y.reindex(X_train.index),
            X_train.loc['close'],
            decay=0.5, num_threads=1)
    elif sample_weights_type == 'none':
        sample_weights = None


###### VIDJET STO S OVIM
# Remove close column if pcais used
# if any("PCA" in s for s in col_names.tolist()):
#     close_index = np.where("close" in s for s in col_names.tolist())
#     np.delete(X_train, close_index, axis=1)
#     X_train = X_train.drop(columns=['close'])
#     X_test = X_test.drop(columns=['close'])
###### VIDJET STO S OVIM


# Cross validation
if cv_type == 'GapWalkForward' or Y.ret.isna().all():
    cv = tscv.GapWalkForward(n_splits=10, gap_size=8, test_size=1)
elif cv_type == 'purged_kfold':
    cv = ml.cross_validation.PurgedKFold(
        n_splits=cv_number,
        samples_info_sets=Y['t1'].reindex(X_train.index))


# Model
clf = RandomForestClassifier(criterion='entropy',
                             # max_features=max_features,
                             min_weight_fraction_leaf=min_weight_fraction_leaf,
                             max_depth=max_depth,
                             n_estimators=n_estimators,
                             class_weight=class_weight,
                             # random_state=rand_state,
                             n_jobs=16)


# Fit model
if cv_type == 'GapWalkForward' or Y.ret.isna().all():
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
elif cv_type == 'purged_kfold':
    scores = ml.cross_validation.ml_cross_val_score(
        clf, X_train, y_train, cv_gen=cv, 
        sample_weight_train=sample_weights,
        scoring=sklearn.metrics.accuracy_score)  #sklearn.metrics.f1_score(average='weighted')


# CV results
mean_score = scores.mean()
std_score = scores.std()
print(f'Mean score: {mean_score}')
writer.add_scalar(tag='mean_score', scalar_value=mean_score, global_step=None)
writer.add_scalar(tag='std_score', scalar_value=std_score, global_step=None)


# retrain the model if mean score is high enough (higher than 0.5)
if mean_score < 0.55:
    print('good_performance: False')
else:
    print('good_performance: True')
    
    # refit the model and get results
    clf = RandomForestClassifier(criterion='entropy',
                                # max_features=max_features,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_depth=max_depth,
                                n_estimators=n_estimators,
                                class_weight=class_weight,
                                # random_state=rand_state,
                                n_jobs=16)
    if not Y['ret'].isna().all():
        clf.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        clf.fit(X_train, y_train)
    tml.modeling.metrics_summary.clf_metrics_tensorboard(
        writer, clf, X_train, X_test, y_train, y_test, avg='binary')

    # Save feature importance tables and plots
    imp = importances(clf, X_test, pd.Series(y_test)) # permutation
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)
    vals = np.abs(shap_values[0]).mean(0)
    shap_importance = pd.Series(vals, index=X_train.columns).rename('shap')
    shap_importance.sort_values(ascending=False, inplace=True)
    imp = pd.concat([imp, shap_importance], axis=1)
    imp.to_csv('feature_importnace.csv')

    # Save model
    import sklearn_json as skljson
    file_name = os.path.join('models', 'random_forest.json')
    clf_ser = serialize_random_forest(clf)
    with open(file_name, 'w') as model_json:
        json.dump(clf_ser, model_json)  
    
    deserialized_model = skljson.from_json(file_name)
    deserialized_model.predict(X_test)
    
        
    # add to dropbox
    # import dropbox
    # dbx = dropbox.Dropbox('RPqFmEm0LbUAAAAAAAAAAZ5Q4ZbVET-HQh18ixMUp6Gcx5lc0vMYMzMA2rueMjO6')
    # with open(file_name, 'rb') as f:
    #     dbx.files_upload(f.read(), '/trend_labeling/' + file_name, mute = True)

    
    ### REFIT THE MODEL WITH MOST IMPORTANT FEATURES
    # fi_cols = shap_values['col_name'].head(keep_important_features)
    # X_train_important = X_train[fi_cols]
    # X_test_important = X_test[fi_cols]
    # clf = RandomForestClassifier(criterion='entropy',
    #                         max_features=keep_important_features,
    #                         min_weight_fraction_leaf=min_weight_fraction_leaf,
    #                         max_depth=max_depth,
    #                         n_estimators=n_estimators,
    #                         class_weight=class_weight,
    #                         # random_state=rand_state,
    #                         n_jobs=16)
    # clf_important = clf.fit(X_train_important, y_train, sample_weight=sample_weights)
    # tml.modeling.metrics_summary.clf_metrics_tensorboard(
    #     writer, clf_important, X_train_important,
    #     X_test_important, y_train, y_test, avg='binary', prefix='fi_')

# close writer
writer.close()


import clr
clr.Add