import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pycaret.classification import *
import sklearn
from sklearn.model_selection import train_test_split
import mlfinlab as ml
import trademl as tml



### IMPORT PREPARED DATA
input_data_path = 'D:/algo_trading_files'
X_train = pd.read_pickle(os.path.join(Path(input_data_path), 'X_train.pkl'))
X_test = pd.read_pickle(os.path.join(Path(input_data_path), 'X_test.pkl'))
y_train = pd.read_pickle(os.path.join(Path(input_data_path), 'y_train.pkl'))
y_test = pd.read_pickle(os.path.join(Path(input_data_path), 'y_test.pkl'))
labeling_info = pd.read_pickle(os.path.join(Path(input_data_path), 'labeling_info.pkl'))


# TRAIN TEST SPLIT
train = pd.concat([X_train, y_train], axis=1)
train['bin'] = train['bin'].astype(np.int)
test = pd.concat([X_test, y_test], axis=1)
test['bin'] = test['bin'].astype(np.int)


# PYCARET SETUP

#intialize the setup
# numeric_features = [col for col in data.columns if col != ['tick_rule']]
exp_clf = setup(train,
                target='bin',
                train_size=0.9,
                categorical_features=['tick_rule'],
                pca=True,
                pca_method='linear',
                pca_components=10,
                remove_perfect_collinearity=True,
                remove_multicollinearity=False,
                # outliers_threshold=None,
                silent=True,  # don't need to confrim data types
                html=False,
                session_id=123)
compare_models(fold=8, turbo=True)

# create models
et = create_model('et')
catboost = create_model('catboost')
lightgbm = create_model('lightgbm')
rf = create_model('rf')
dt = create_model('dt')
gbc = create_model('gbc')
xgboost = create_model('xgboost')
knn = create_model('knn')

# tune models
tuned_et = tune_model(et)
tuned_catboost = tune_model(catboost)
tuned_lightgbm = tune_model(lightgbm)
tunrd_rf = tune_model(rf)
tuned_dt = tune_model(dt)
tuned_gbc = tune_model(gbc)
tuned_xgboost = tune_model(xgboost)
tuned_knn = tune_model(knn)

# blend trained models
blend_specific = blend_models(
    estimator_list = [tuned_et, tuned_lightgbm, tunrd_rf, tuned_dt, tuned_gbc, tuned_xgboost, tuned_knn]
    )

# stack trained models
stacked_models = stack_models(
    estimator_list = [tuned_et, tuned_lightgbm, tunrd_rf, tuned_dt, tuned_gbc, tuned_xgboost, tuned_knn]
    )


# plot models
# plot_model(tuned_et, plot = 'auc')
# plot_model(tuned_et, plot = 'pr')
# plot_model(tuned_et, plot = 'feature')
# plot_model(tuned_et, plot = 'confusion_matrix')

# predictions
predict_model(tuned_et)
predict_model(tuned_catboost)
predict_model(tuned_lightgbm)
predict_model(tunrd_rf)
predict_model(tuned_dt)
predict_model(tuned_gbc)
predict_model(tuned_xgboost)
predict_model(tuned_knn)
predict_model(blend_specific)

# finalize the model
final_et = finalize_model(tuned_et)
final_catboost = finalize_model(tuned_catboost)
final_lightgbm = finalize_model(tuned_lightgbm)
final_rf = finalize_model(tunrd_rf)
final_dt = finalize_model(tuned_dt)
final_gbc = finalize_model(tuned_gbc)
final_xgboost = finalize_model(tuned_xgboost)
final_knn = finalize_model(tuned_knn)
final_blend_specific = finalize_model(blend_specific)

# predict on unseen dataset
unseen_predictions_et = predict_model(final_et, data=test)
unseen_predictions_et.head()

# predict on unseen dataset
predictions_et = predict_model(final_et, data=test)
predictions_catboost = predict_model(final_catboost, data=test)
predictions_lightgbm = predict_model(final_lightgbm, data=test)
predictions_rf = predict_model(final_rf, data=test)
predictions_dt = predict_model(final_dt, data=test)
predictions_gbc = predict_model(final_gbc, data=test)
predictions_xgboost = predict_model(final_xgboost, data=test)
predictions_knn = predict_model(final_knn, data=test)
predictions_blend = predict_model(final_blend_specific, data=test)

# accuracy scores
sklearn.metrics.accuracy_score(unseen_predictions_et['bin'], unseen_predictions_et['Label'])
sklearn.metrics.accuracy_score(predictions_catboost['bin'], predictions_catboost['Label'])
sklearn.metrics.accuracy_score(predictions_lightgbm['bin'], predictions_lightgbm['Label'])
sklearn.metrics.accuracy_score(predictions_rf['bin'], predictions_rf['Label'])
sklearn.metrics.accuracy_score(predictions_dt['bin'], predictions_dt['Label'])
sklearn.metrics.accuracy_score(predictions_gbc['bin'], predictions_gbc['Label'])
sklearn.metrics.accuracy_score(predictions_xgboost['bin'], predictions_xgboost['Label'])
sklearn.metrics.accuracy_score(predictions_knn['bin'], predictions_knn['Label'])
sklearn.metrics.accuracy_score(predictions_blend['bin'], predictions_blend['Label'])

# save models
save_model(final_et,'final_et_model')
save_model(final_catboost,'final_catboost_model')
save_model(final_lightgbm,'final_lightgbm_model')
save_model(final_rf,'final_rf_model')
save_model(final_dt,'final_dt_model')
save_model(final_gbc,'final_gbc_model')
save_model(final_xgboost,'final_xgboost_model')
save_model(final_knn,'final_knn_model')
save_model(final_blend_specific,'final_blend_model')

# load models
final_et = load_model('final_et_model')
final_catboost = load_model('final_catboost_model')
final_lightgbm = load_model('final_lightgbm_model')
final_rf = load_model('final_rf_model')
final_dt = load_model('final_dt_model')
final_gbc = load_model('final_gbc_model')
final_xgboost = load_model('final_xgboost_model')
final_knn = load_model('final_knn_model')
final_blend_specific = load_model('final_blend_model')






# from sktime.datasets import load_arrow_head


# X, y = load_arrow_head(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# X.shape