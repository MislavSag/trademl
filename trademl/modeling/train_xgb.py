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
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier
from sklearn.base import clone
import shap
import xgboost as xgb
import mlfinlab as ml
from mlfinlab.feature_importance import get_orthogonal_features
import trademl as tml
from tensorboardX import SummaryWriter
matplotlib.use("Agg")  # don't show graphs


### TENSORBORADX WRITER
log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)


### MODEL HYPERPARAMETERS
input_data_path = 'D:/algo_trading_files'
use_pca_features = False
rand_state = 3
sample_weights_type = 'return'
cv_type = 'purged_kfold'
cv_number = 5
# model
booster = 'gbtree'
# num_feature = 15
eta = 0.2
min_child_weight = 1
subsample = 0.95
colsample_bytree = 0.9
max_depth = 3
learning_rate = 0.1


### IMPORT PREPARED DATA
if use_pca_features:
    X_train = pd.read_pickle(os.path.join(Path(input_data_path), 'X_train_pca.pkl'))
    X_test = pd.read_pickle(os.path.join(Path(input_data_path), 'X_test_pca.pkl'))
    y_train = pd.read_pickle(os.path.join(Path(input_data_path), 'y_train_pca.pkl'))
    y_test = pd.read_pickle(os.path.join(Path(input_data_path), 'y_test_pca.pkl'))
    labeling_info = pd.read_pickle(os.path.join(Path(input_data_path), 'labeling_info_pca.pkl'))
else:
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


# MODEL
# convert pandas df to xgboost matrix
# dmatrix_train = xgb.DMatrix(data=X_train, label=y_train.astype(int))
# dmatrix_test = xgb.DMatrix(data=X_test, label=y_test.astype(int))

# # parameters for GridSearch
# params = {
#     'booster': booster,
#     # 'num_feature': num_feature,
#     'eta': eta,
#     'min_child_weight': min_child_weight,
#     'subsample': subsample,
#     'max_depth': max_depth,
#     'learning_rate': learning_rate,
#     'colsample_bytree': colsample_bytree
#     }

# # define estimator
# bst = xgb.train(params, dmatrix_train)
# # make prediction
# preds = bst.predict(dmatrix_test)

# # cv
# cv_results = xgb.cv(
#     params=params,
#     dtrain=dmatrix_train,
#     num_boost_round = 999,  # Here we will use a large number again and count on early_stopping_rounds to find the optimal number of rounds before reaching the maximum.
#     folds=cv,
#     early_stopping_rounds=10,
#     metrics="auc",
#     as_pandas=True
#     )

# split X_train data
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train.astype(int),
    test_size=0.15, shuffle=False, stratify=None)
sample_weights_train=sample_weights.iloc[:X_train.shape[0]]

clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=max_depth,
    learning_rate=learning_rate,
    verbosity=1,
    objective='binary:logistic',
    booster=booster,
    njobs=16,
    min_child_weight=min_child_weight,
    subsample=subsample,
    colsample_bytree=colsample_bytree
)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc", early_stopping_rounds=30, sample_weight=sample_weights_train, verbose=True)


### CV RESULTS
# eval scores
evals_result = clf.evals_result()['validation_0']
evals_result = np.array(list(evals_result.values())).reshape(-1)
mean_score = evals_result.mean()
best_score = evals_result.max()
std_score = evals_result.std()
save_id = f'{max_depth}{learning_rate}{min_child_weight}{str(mean_score)[2:6]}'
print(f'Mean score: {best_score}')
writer.add_scalar(tag='mean_score', scalar_value=mean_score, global_step=None)
writer.add_scalar(tag='std_score', scalar_value=std_score, global_step=None)
writer.add_scalar(tag='best_score', scalar_value=std_score, global_step=None)
writer.add_text(tag='save_id', text_string=save_id, global_step=None)


if best_score > 0.55:
    # test scores
    clf_predictions = clf.predict(X_test)
    tml.modeling.metrics_summary.clf_metrics_tensorboard(
        writer, clf, X_train, X_test, y_train, y_test, avg='binary')

    # save feature importance tables and plots
    # xgb.plot_importance(clf)
    
    
    # shap_values, importances, mdi_feature_imp = tml.modeling.feature_importance.important_fatures(
    #     clf, X_train, y_train, plot_name=save_id)
    # tml.modeling.utils.save_files([shap_values, importances, mdi_feature_imp],
    #             file_names=[f'shap_{save_id}.csv',
    #                         f'rf_importance_{save_id}.csv',
    #                         f'mpi_{save_id}.csv'],
    #             directory=os.path.join(Path(input_data_path), 'important_features'))