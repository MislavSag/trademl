# fundamental modules
import numpy as np
import scipy
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import pyplot
from IPython.display import display
# preprocessing
from sklearn.model_selection import train_test_split
# modelling
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV, KFold
# from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier
import xgboost
from sacred import Experiment
# metrics 
import mlfinlab as ml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    log_loss
    )
# features engeenering
# from mlfinlab.feature_importance import (
#     mean_decrease_impurity,
#     mean_decrease_accuracy,
#     single_feature_importance,
#     plot_feature_importance,
#     get_orthogonal_features,
# )
# from mlfinlab.structural_breaks import (
#     get_chu_stinchcombe_white_statistics,
#     get_chow_type_stat, get_sadf)
# finance packages
# from mlfinlab.sample_weights import (
#     get_weights_by_return,
#     get_weights_by_time_decay
#     )
# from mlfinlab.cross_validation import PurgedKFold, ml_cross_val_score
# from mlfinlab.backtest_statistics import timing_of_flattening_and_flips
# from mlfinlab.bet_sizing import (
#     bet_size_probability, bet_size_dynamic, bet_size_budget, bet_size_reserve,
#     confirm_and_cast_to_df, get_concurrent_sides, cdf_mixture,
#     single_bet_size_mixed, M2N, centered_moment, raw_moment, 
#     most_likely_parameters
#     )
# from mlfinlab.backtest_statistics import average_holding_period
# backtesting
# import  mlfinlab.backtest_statistics as bs
# from  mlfinlab.backtest_statistics import  information_ratio
import pyfolio as pf
# other
import trademl
import  trademl.modeling as ms
from sklearn.base import clone


### GLOBAL (CONFIGS)

DATA_PATH = Path('C:/Users/Mislav/algoAItrader/data/spy_store_stat.h5')


### IMPORT AND ADD FEATURES

with pd.HDFStore(DATA_PATH) as store:
    spy = store.get('spy_store_stat')
    

### TRIPLE-BARRIER LABELING

# Compute daily volatility
daily_vol = ml.util.get_daily_vol(close=spy['close_orig'], lookback=50)

# Apply Symmetric CUSUM Filter and get timestamps for events
cusum_events = ml.filters.cusum_filter(spy['close_orig'],
                                        threshold=daily_vol.mean()*0.5)

# Compute vertical barrier
vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events,
                                                        close=spy['close_orig'],
                                                        num_days=4200)

if __name__ == '__main__':   
    # make triple barriers (if side_prediction arg is omitted, return -1, 0, 1 
    # (w/t which touched first))
    pt_sl = [2, 2]  # IF ONLY SECONDARY (ML) MODEL HORIZONTAL BARRIERS SYMMETRIC!
    min_ret = 0.004
    triple_barrier_events = ml.labeling.get_events(
        close=spy['close_orig'],
        t_events=cusum_events,
        pt_sl=pt_sl,
        target=daily_vol,
        min_ret=min_ret,
        num_threads=4,
        vertical_barrier_times=vertical_barriers)
    display(triple_barrier_events.head(10))

# labels
labels = ml.labeling.get_bins(triple_barrier_events, spy['close_orig'])
display(labels.head(10))
display(labels.bin.value_counts())
labels = ml.labeling.drop_labels(labels)
triple_barrier_events = triple_barrier_events.reindex(labels.index)


### PREPARE MOEL

# get data at triple-barrier evetns
X = spy.drop(columns=['close_orig']).reindex(labels.index)  # PROVJERITI OVO. MODA IPAK IDE TRIPPLE-BARRIER INDEX ???
y = labels['bin']
print('X shape: ', X.shape); print('y shape: ', y.shape)
print('y counts:\n', y.value_counts())

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.10, 
                                                    shuffle=False, 
                                                    stratify=None)
print(X_train.shape); print(y_train.shape)
print(X_test.shape); print(y_test.shape)


### SAMPLE WEIGHTS

return_sample_weights = get_weights_by_return(
    triple_barrier_events.loc[X_train.index],
    spy.loc[X_train.index, 'close_orig'],
    num_threads=1)
time_sample_weights = get_weights_by_time_decay(
    triple_barrier_events.loc[X_train.index],
    spy.loc[X_train.index, 'close_orig'],
    decay=0.5, num_threads=1)


### ML MODEL

# RF GRID CV

# parameters for GridSearch
parameters = {'max_depth': [2, 3, 4, 5, 10],
                'n_estimators': [600, 1000, 1400]}

# CV generators
cv_gen_purged = PurgedKFold(
    n_splits=4,
    samples_info_sets=triple_barrier_events.reindex(X_train.index).t1)
rf = RandomForestClassifier(criterion='entropy',
                            max_features=10,
                            min_weight_fraction_leaf=0.05,
                            class_weight='balanced_subsample')
clf = GridSearchCV(rf,
                    param_grid=parameters,
                    scoring='f1',
                    n_jobs=16,
                    cv=cv_gen_purged)
clf.fit(X_train, y_train, sample_weight=return_sample_weights)
depth, n_estimators = clf.best_params_.values()
rf_best = RandomForestClassifier(criterion='entropy',
                                    max_features=1,
                                    min_weight_fraction_leaf=0.05,
                                    max_depth=depth,
                                    n_estimators=n_estimators,
                                    class_weight='balanced_subsample')
rf_best.fit(X_train, y_train, sample_weight=return_sample_weights)
trademl.metrics_summary.display_clf_metrics(
    rf_best, X_train, X_test, y_train, y_test)

display_clf_metrics(rf_best, X_train, X_test, y_train, y_test)
plot_roc_curve(rf_best, X_train, X_test, y_train, y_test)

# SequentiallyBootstrappedBaggingClassifier
# if __name__ == '__main__':   
base_est = RandomForestClassifier(
    n_estimators=1, criterion='entropy',
    bootstrap=False, class_weight='balanced_subsample')
clf = SequentiallyBootstrappedBaggingClassifier(
    base_estimator=base_est,
    samples_info_sets=triple_barrier_events.t1,
    price_bars=price_bars, oob_score=True)


### FEATURE IMPORTANCE

# mean decreasing impurity
mdi_feature_imp = ml.feature_importance.mean_decrease_impurity(
    rf_best, X_train.columns)
ml.feature_importance.plot_feature_importance(
    mdi_feature_imp, 0, 0, save_fig=True,
    output_path='features/mdi_feat_imp.png')

# mean decreasing accuracy
mda_feature_imp = ml.feature_importance.mean_decrease_accuracy(
    rf_best, X_train, y_train, cv_gen_purged,
    scoring=log_loss,
    sample_weight=return_sample_weights.values)
plot_feature_importance(mda_feature_imp, 0, 0, save_fig=True,
                        output_path='features/mda_feat_imp.png')

# single feature importance
rf_best_ = clone(rf_best)  # seems sfi change learner somehow, and can't use it later
sfi_feature_imp = ml.feature_importance.single_feature_importance(
    rf_best_, X_train, y_train, cv_gen_purged,
    scoring=accuracy_score,
    sample_weight=return_sample_weights.values)
plot_feature_importance(sfi_feature_imp, 0, 0, save_fig=True,
                        output_path='features/sfi_feat_imp.png')

# clustered feature importance algorithm