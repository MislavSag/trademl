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
from sklearn.base import clone
import shap
import mlfinlab as ml
from mlfinlab.feature_importance import get_orthogonal_features
import trademl as tml
from tensorboardX import SummaryWriter
import uuid
matplotlib.use("Agg")  # don't show graphs


### TENSORBORADX WRITER
log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)


### MODEL HYPERPARAMETERS
input_data_path = Path('D:/algo_trading_files')
sample_weights_type = 'return'
cv_type = 'purged_kfold'
cv_number = 5
max_depth = 2
max_features = 10
n_estimators = 350
min_weight_fraction_leaf = 0.1  
class_weight = 'balanced_subsample'
keep_important_features = 20  # for feature selection


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


### MODEL
clf = RandomForestClassifier(criterion='entropy',
                                # max_features=max_features,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_depth=max_depth,
                                n_estimators=n_estimators,
                                class_weight=class_weight,
                                # random_state=rand_state,
                                n_jobs=16)
scores = ml.cross_validation.ml_cross_val_score(
    clf, X_train, y_train, cv_gen=cv, 
    sample_weight_train=sample_weights,
    scoring=sklearn.metrics.accuracy_score)  #sklearn.metrics.f1_score(average='weighted')


### CV RESULTS
mean_score = scores.mean()
std_score = scores.std()
save_id = str(uuid.uuid1())
# save_id = f'{max_depth}{max_features}{n_estimators}{str(mean_score)[2:6]}'
print(f'Mean score: {mean_score}')
writer.add_scalar(tag='mean_score', scalar_value=mean_score, global_step=None)
writer.add_scalar(tag='std_score', scalar_value=std_score, global_step=None)
writer.add_text(tag='save_id', text_string=save_id, global_step=None)

# retrain the model if mean score is high enough (higher than 0.5)
if mean_score < 0.55:
    print('good_performance: False')
else:
    print('good_performance: True')
    
    # refit the model and get results
    clf = RandomForestClassifier(criterion='entropy',
                                max_features=max_features,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_depth=max_depth,
                                n_estimators=n_estimators,
                                class_weight=class_weight,
                                # random_state=rand_state,
                                n_jobs=16)
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    tml.modeling.metrics_summary.clf_metrics_tensorboard(
        writer, clf, X_train, X_test, y_train, y_test, avg='binary')

    # save feature importance tables and plots
    shap_values, importances, mdi_feature_imp = tml.modeling.feature_importance.important_features(
        clf, X_train, y_train, save_id,
        save_path=os.path.join(Path(input_data_path), 'fi_plots'))
    tml.modeling.utils.save_files([shap_values, importances, mdi_feature_imp],
                file_names=[f'shap_{save_id}.csv',
                            f'rf_importance_{save_id}.csv',
                            f'mpi_{save_id}.csv'],
                directory=os.path.join(Path(input_data_path), 'important_features'))
    
    
    # ### REFIT THE MODEL WITH MOST IMPORTANT FEATURES
    fi_cols = shap_values['col_name'].head(keep_important_features)
    X_train_important = X_train[fi_cols]
    X_test_important = X_test[fi_cols]
    clf = RandomForestClassifier(criterion='entropy',
                            max_features=keep_important_features,
                            min_weight_fraction_leaf=min_weight_fraction_leaf,
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            class_weight=class_weight,
                            # random_state=rand_state,
                            n_jobs=16)
    clf_important = clf.fit(X_train_important, y_train, sample_weight=sample_weights)
    tml.modeling.metrics_summary.clf_metrics_tensorboard(
        writer, clf_important, X_train_important,
        X_test_important, y_train, y_test, avg='binary', prefix='fi_')

# close writer
writer.close()
