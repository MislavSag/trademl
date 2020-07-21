# fundamental modules
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
from pathlib import Path
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier
from sklearn.base import clone
import xgboost
import shap
import mlfinlab as ml
import trademl as tml

### DON'T SHOW GRAPH OPTION
matplotlib.use("Agg")


### GLOBALS
DATA_PATH = 'D:/market_data/usa/ohlcv_features'

### NON-MODEL HYPERPARAMETERS
num_threads = 1
structural_break_regime = 'all'
labeling_technique = 'trend_scanning'
std_outlier = 10
tb_volatility_lookback = 500
tb_volatility_scaler = 1
tb_triplebar_num_days = 10
tb_triplebar_pt_sl = [1, 1]
tb_triplebar_min_ret = 0.004
ts_look_forward_window = 1200  # 60 * 8 * 10 (10 days)
ts_min_sample_length = 30
ts_step = 5
tb_min_pct = 0.10
sample_weights_type = 'returns'
cv_type = 'purged_kfold'
cv_number = 4
rand_state = 3
stationary_close_lables = False

### MODEL HYPERPARAMETERS
max_depth = 3
max_features = 10
n_estimators = 500
min_weight_fraction_leaf = 0.05
class_weight = 'balanced_subsample'

### FEATURE SELECTION
keep_important_features = 20


def import_data(data_path, remove_cols, contract='SPY'):
    # import data
    with pd.HDFStore(data_path + '/' + contract + '.h5') as store:
        data = store.get(contract)
    data.sort_index(inplace=True)
    
    # remove variables
    remove_cols = [col for col in remove_cols if col in data.columns]
    data.drop(columns=remove_cols, inplace=True)
    
    return data


if __name__ == '__main__':
    
    ### IMPORT DATA
    remove_ohl = ['open', 'low', 'high', 'average', 'barCount',
                  'open_vix', 'high_vix', 'low_vix', 'close_vix', 'volume_vix',
                  'open_orig', 'high_orig', 'low_orig']
    data = import_data(DATA_PATH, remove_ohl, contract='SPY')


    ### REGIME DEPENDENT ANALYSIS
    if structural_break_regime == 'chow':
        if (data.loc[data['chow_segment'] == 1].shape[0] / 60 / 8) < 365:
            data = data.iloc[-(60*8*365):]
        else:
            data = data.loc[data['chow_segment'] == 1]

    ### USE STATIONARY CLOSE TO CALCULATE LABELS
    if stationary_close_lables:
        data['close_orig'] = data['close']  # with original close reslts are pretty bad!


    ### LABELING
    if labeling_technique == 'triple_barrier':
        # TRIPLE BARRIER LABELING
        triple_barrier_pipe= tml.modeling.pipelines.TripleBarierLabeling(
            close_name='close_orig',
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
    elif labeling_technique == 'trend_scanning':
        trend_scanning_pipe = tml.modeling.pipelines.TrendScanning(
            close_name='close_orig',
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
        labeling_info = ml.labeling.fixed_time_horizon(data['close_orig'], threshold=0.005, resample_by='B').dropna().to_frame()
        labeling_info = labeling_info.rename(columns={'close_orig': 'bin'})
        print(labeling_info.iloc[:, 0].value_counts())
        X = X.iloc[:-1, :]


    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(columns=['close_orig']), labeling_info['bin'],
        test_size=0.10, shuffle=False, stratify=None)


    ### SAMPLE WEIGHTS (DECAY FACTOR CAN BE ADDED!)
    if sample_weights_type == 'returns':
        sample_weigths = ml.sample_weights.get_weights_by_return(
            labeling_info.reindex(X_train.index),
            data.loc[X_train.index, 'close_orig'],
            num_threads=1)
    elif sample_weights_type == 'time_decay':
        sample_weigths = ml.sample_weights.get_weights_by_time_decay(
            labeling_info.reindex(X_train.index),
            data.loc[X_train.index, 'close_orig'],
            decay=0.5, num_threads=1)
    elif labeling_technique is 'trend_scanning':
        sample_weigths = labeling_info['t_value'].reindex(X_train.index).abs()
    elif labeling_technique is 'none':
        sample_weigths = None


    ### CROS VALIDATION STEPS
    if cv_type == 'purged_kfold':
        cv = ml.cross_validation.PurgedKFold(
            n_splits=cv_number,
            samples_info_sets=labeling_info['t1'].reindex(X_train.index))


    ### MODEL
    # MLDP str 98/99
    clf = RandomForestClassifier(criterion='entropy',
                                 max_features=max_features,
                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                                 max_depth=max_depth,
                                 n_estimators=n_estimators,
                                 class_weight=class_weight,
                                 random_state=rand_state,
                                 n_jobs=16)
    scores = ml.cross_validation.ml_cross_val_score(
        clf, X_train, y_train, cv_gen=cv, 
        sample_weight_train=sample_weigths,
        scoring=sklearn.metrics.accuracy_score)  #sklearn.metrics.f1_score(average='weighted')
    mean_score = scores.mean()
    std_score = scores.std()
    save_id = f'{max_depth}{max_features}{n_estimators}{str(mean_score)[2:6]}'

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
                                    random_state=rand_state,
                                    n_jobs=16)
        clf.fit(X_train, y_train, sample_weight=sample_weigths)
        tml.modeling.metrics_summary.clf_metrics(
            clf, X_train, X_test, y_train, y_test, avg='binary')


        # features seleciton
        def feature_importance_values(clf, X_train, y_train, plot_name):

            # clone clf to not change it
            clf_ = sklearn.clone(clf)
            clf_.fit(X_train, y_train)

            # SHAP values
            explainer = shap.TreeExplainer(model=clf_, model_output='raw')
            shap_values = explainer.shap_values(X_train)
            vals= np.abs(shap_values).mean(0)
            feature_importance = pd.DataFrame(
                list(zip(X_train.columns, sum(vals))),
                columns=['col_name','feature_importance_vals'])
            feature_importance.sort_values(
                by=['feature_importance_vals'], ascending=False, inplace=True)
            
            # save plots
            # create directory if it does not exists
            if not os.path.exists('plots'):
                os.makedirs('plots')
            saving_path = Path(f'plots/shap_{plot_name}.png')
                    
            # shap plot
            shap.summary_plot(shap_values, X_train,
                            plot_type='bar', max_display=15,
                            show=False)
            plt.savefig(saving_path)
            
            # random forest default feature importance
            importances = pd.Series(clf_.feature_importances_, index=X_train.columns)
            importances = importances.sort_values(ascending=False)
            
            # mean decreasing impurity
            mdi_feature_imp = ml.feature_importance.mean_decrease_impurity(
                clf_, X_train.columns)
            
            # mean decreasing accuracy (COMMENT DUE TO EFFICIENCY PROBLEM)
            # mda_feature_imp = ml.feature_importance.mean_decrease_accuracy(
            #     clf_, X_train, y_train, cv, scoring=log_loss,
            #     sample_weight_train=sample_weigths.values)

            # base estimator for mlfinlab features importance (COMMENT DUE TO EFFICIENCY PROBLEM)
            # rf_best_one_features = RandomForestClassifier(
            #     criterion='entropy',
            #     max_features=1,
            #     min_weight_fraction_leaf=0.05,
            #     max_depth=max_depth,
            #     n_estimators=n_estimators,
            #     max_leaf_nodes=max_leaf_nodes,
            #     class_weight='balanced',
            #     n_jobs=16)
            
            # # doesn't work?
            # sfi_feature_imp = ml.feature_importance.single_feature_importance(
            #     rf_best_one_features, X_train, y_train, cv,
            #     scoring=sklearn.metrics.accuracy_score,
            #     sample_weight_train=sample_weigths.values)
            
            return feature_importance, importances, mdi_feature_imp


        def save_files(objects, file_names, directory='important_features'):            
            # create directory if it does not exists
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # save files to directory
            for df, file_name in zip(objects, file_names):
                saving_path = Path(f'{directory}/{file_name}')
                if ".csv" not in file_names: 
                    df.to_csv(saving_path)


        # save feature importance tables and plots
        # tml.modeling.feature_importance.feature_importance(
        #     clf, X_train, y_train, plot_name=save_id)
        shap_values, importances, mdi_feature_imp = feature_importance_values(
            clf, X_train, y_train, plot_name=save_id)
        save_files([shap_values, importances, mdi_feature_imp],
                   file_names=[f'shap_{save_id}.csv',
                               f'rf_importance_{save_id}.csv',
                               f'mpi_{save_id}.csv'],
                   directory='important_features')
        
        
        # ### REFIT THE MODEL WITH MOST IMPORTANT FEATURES
        fi_cols = shap_values['col_name'].head(keep_important_features)
        X_train_important = X_train[fi_cols]
        X_test_important = X_test[fi_cols]
        clf_important = clf.fit(X_train_important, y_train)
        tml.modeling.metrics_summary.clf_metrics(
            clf_important, X_train_important,
            X_test_important, y_train, y_test, avg='binary', prefix='fi_')
