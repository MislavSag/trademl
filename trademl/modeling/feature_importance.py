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


def important_fatures(clf, X_train, y_train, plot_name):

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
