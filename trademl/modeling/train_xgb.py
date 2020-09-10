from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import shap
import xgboost as xgb
import mlfinlab as ml
import trademl as tml
from tensorboardX import SummaryWriter
import random
import pickle
matplotlib.use("Agg")  # don't show graphs



### TENSORBORADX WRITER
log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)


### MODEL HYPERPARAMETERS
input_data_path = Path('D:/algo_trading_files')
save_model = False
rand_state = 3
sample_weights_type = 'return'
cv_type = 'purged_kfold'
cv_number = 5
# model
booster = 'gbtree'
eta = 0.2
min_child_weight = 5
subsample = 0.75
colsample_bytree = 0.9
max_depth = 3
learning_rate = 0.09


### IMPORT PREPARED DATA
X_train = pd.read_pickle(os.path.join(Path(input_data_path), 'X_train.pkl'))
X_test = pd.read_pickle(os.path.join(Path(input_data_path), 'X_test.pkl'))
y_train = pd.read_pickle(os.path.join(Path(input_data_path), 'y_train.pkl'))
y_test = pd.read_pickle(os.path.join(Path(input_data_path), 'y_test.pkl'))
labeling_info = pd.read_pickle(os.path.join(Path(input_data_path), 'labeling_info.pkl'))


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


### CROS VALIDATION STEPS
if cv_type == 'purged_kfold':
    cv = ml.cross_validation.PurgedKFold(
        n_splits=cv_number,
        samples_info_sets=labeling_info['t1'].reindex(X_train.index))


# MODEL
# convert pandas df to xgboost matrix
dmatrix_train = xgb.DMatrix(data=X_train, label=y_train.astype(int))
dmatrix_test = xgb.DMatrix(data=X_test, label=y_test.astype(int))

# parameters for GridSearch
params = {
    'booster': booster,
    'eta': eta,
    'min_child_weight': min_child_weight,
    'subsample': subsample,
    'max_depth': max_depth,
    'learning_rate': learning_rate,
    'colsample_bytree': colsample_bytree
    }

# cv
cv_clf = xgb.cv(
    params=params,
    dtrain=dmatrix_train,
    num_boost_round = 500,  # Here we will use a large number again and count on early_stopping_rounds to find the optimal number of rounds before reaching the maximum.
    folds=cv,
    early_stopping_rounds=15,
    metrics="auc",
    as_pandas=True
    )

# cv scores
cv_results = cv_clf.loc[cv_clf.iloc[:, 2] == cv_clf.iloc[:, 2].max()]
mean_score = cv_results.iloc[0, 2]
std_score = cv_results.iloc[0, 3]
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
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc',
            early_stopping_rounds=30, sample_weight=sample_weights_train, verbose=True)

    # eval scores
    evals_result = clf.evals_result()['validation_0']
    evals_result = np.array(list(evals_result.values())).reshape(-1)
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
        tml.modeling.feature_importance.fi_shap(clf, X_train, y_train, save_id, input_data_path)
        tml.modeling.feature_importance.fi_xgboost(clf, X_train, save_id, input_data_path)

# save model
if save_model:
    pickle.dump(clf, open(os.path.join(Path(input_data_path), 'good_models', "xgboost.dat"), "wb"))
