import pandas as pd
import numpy as np
import os
from pathlib import Path
import sklearn
import shap
import matplotlib.pyplot as plt
import mlfinlab as ml


def feature_importance_values(clf, X_train, y_train):

    # clone clf to not change it
    clf_ = sklearn.clone(clf)
    clf_.fit(X_train, y_train)

    # SHAP values
    explainer = shap.TreeExplainer(model=clf_, model_output='raw')
    shap_values = explainer.shap_values(X_train)

    return shap_values


def feature_importnace_vec(shap_val, X_train):
    # SHAP values
    vals= np.abs(shap_val).mean(0)
    feature_importance = pd.DataFrame(
        list(zip(X_train.columns, sum(vals))),
        columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(
        by=['feature_importance_vals'], ascending=False, inplace=True)
    return feature_importance


def plot_feature_importance(shap_val, X_train, name):
    # SHAP values
    shap.initjs()
    shap.summary_plot(shap_val, X_train, plot_type='bar', max_display=25,
                      show=False)
    plt.savefig(f'{name}feature_importance.png')


def important_features(clf, X_train, y_train, plot_name, save_path):

    # clone clf to not change it
    clf_ = sklearn.clone(clf)
    clf_.fit(X_train, y_train)

    # SHAP values
    explainer = shap.TreeExplainer(model=clf_)
    shap_values = explainer.shap_values(X_train)
    vals= np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(
        list(zip(X_train.columns, sum(vals))),
        columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(
        by=['feature_importance_vals'], ascending=False, inplace=True)
    
    # save plots
    # create directory if it does not exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saving_path = os.path.join(Path(save_path),f'shap_{plot_name}.png')
            
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


def fi_shap(clf, X_train, y_train, save_id, save_path):
    """Function calculate SHAP values, save plots and tables of most important features:

    Args:
        clf (xgb.sklearn.XGBClassifier or ): Classifier for which tto calculate important features.
        X_train (pd.DataFrame): X_train data used to calculate again (probabbly can be overcomed somehow)
        y_train (pd.DataFrame): y_train data used to calculate again (probabbly can be overcomed somehow)
        plot_name (str): used as part of the plot name
        save_path (str): path to directory where plots and tables are saved

    Returns:
        [type]: feature_importance, importances, mdi_feature_imp
    """
    # define and create directories
    save_path = Path(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path_plots = os.path.join(save_path, 'fi_plots')
    if not os.path.exists(save_path_plots):
        os.makedirs(save_path_plots)
    save_path_tables = os.path.join(save_path, 'important_features')
    if not os.path.exists(save_path_tables):
        os.makedirs(save_path_tables)

    ######## THIS PART WOULD BE PROBABLE DELTE LATER AFTER THEY SOLVE ISSUE: https://github.com/slundberg/shap/issues/1215 ########
    if isinstance(clf, xgb.sklearn.XGBClassifier):
        clf_ = clf.get_booster()
        model_bytearray = clf_.save_raw()[4:]
        def myfun(self=None):
            return model_bytearray
        clf_.save_raw = myfun
    ######## THIS PART WOULD BE PROBABLE DELTE LATER AFTER THEY SOLVE ISSUE: https://github.com/slundberg/shap/issues/1215 ########
    else:
        ############# IA M NOT SURE THIS I SNECESSARY ##############
        # clone clf to not change it
        clf_ = sklearn.clone(clf)
        clf_.fit(X_train, y_train)
        ############# IA M NOT SURE THIS I SNECESSARY ##############

    # calculate shap values
    explainer = shap.TreeExplainer(model=clf_)
    shap_values = explainer.shap_values(X_train)
    vals = np.abs(shap_values).mean(0)
    if len(vals.shape) == 1:
        fi = pd.DataFrame(
            list(zip(X_train.columns, vals)),
            columns=['col_name','feature_importance_vals'])
    elif len(vals.shape) == 2:
        fi = pd.DataFrame(
            list(zip(X_train.columns, sum(vals))),
            columns=['col_name','feature_importance_vals'])
    fi.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    
    # save shap plot
    shap.summary_plot(shap_values, X_train,
                        plot_type='bar', max_display=15,
                        show=False)
    plt.savefig(os.path.join(save_path_plots, f'shap_{save_id}.png'))
    
    # save values
    tml.modeling.utils.save_files([fi], [f'shap_{save_id}.csv'], save_path_tables)
    
    return 'Plot and table with shap values saved'


def fi_xgboost(clf, X_train, save_id, save_path):
    """Function calculate SHAP values, save plots and tables of most important features:

    Args:
        clf (xgb.sklearn.XGBClassifier or ): Classifier for which tto calculate important features.
        X_train (pd.DataFrame): X_train data used to calculate again (probabbly can be overcomed somehow)
        plot_name (str): used as part of the plot name
        save_path (str): path to directory where plots and tables are saved

    Returns:
        [type]: feature_importance, importances, mdi_feature_imp
    """
    # assert data types
    assert(clf, xgb.sklearn.XGBClassifier)
        
    # define and create directories
    save_path = Path(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path_plots = os.path.join(save_path, 'fi_plots')
    if not os.path.exists(save_path_plots):
        os.makedirs(save_path_plots)
    save_path_tables = os.path.join(save_path, 'important_features')
    if not os.path.exists(save_path_tables):
        os.makedirs(save_path_tables)
        
    # save plot and tables
    saving_path = os.path.join(save_path_plots,f'xgboost_{plot_name}.png')
    xgb.plot_importance(clf, max_num_features=10)
    plt.savefig(saving_path)
    importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    
    return 'Plot and table with xgboost default values saved'
