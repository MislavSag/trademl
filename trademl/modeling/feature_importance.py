import pandas as pd
import numpy as np
import sklearn


def feature_importance(clf, X_train, y_train):

    # clone clf to not change it
    clf_ = sklearn.clone(clf)
    clf_.fit(X_train, y_train)

    # SHAPE values
    explainer = shap.TreeExplainer(model=clf_, model_output='raw')
    shap_values = explainer.shap_values(X_train)

    return shap_values


def feature_importnace_vec(shap_val):
    # SHAP values
    vals= np.abs(shap_val).mean(0)
    feature_importance = pd.DataFrame(
        list(zip(X_train.columns, sum(vals))),
        columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(
        by=['feature_importance_vals'], ascending=False, inplace=True)
    return feature_importance


def plot_feature_importance(shap_val):
    # SHAP values
    shap.initjs()
    shap.summary_plot(shap_val, X_train, plot_type='bar', max_display=25)
