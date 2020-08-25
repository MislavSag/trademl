from pathlib import Path
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import mlfinlab as ml
from mlfinlab.feature_importance import get_orthogonal_features
import trademl as tml
import mfiles
from dotenv import load_dotenv


### GLOBALS (path to partialy preprocessed data)
DATA_PATH = 'D:/market_data/usa/ohlcv_features/'


### NON-MODEL HYPERPARAMETERS (for guildai)
output_path = 'C:/Users/Mislav/Documents/GitHub/trademl/trademl/modeling/'
num_threads = 1
label = 'day_10'
structural_break_regime = 'all'
labeling_technique = 'trend_scanning'
tb_volatility_lookback = 500
tb_volatility_scaler = 1
tb_triplebar_num_days = 10
sample_weights_type = 'returns'
correlation_threshold = 0.98
pca = False

### MODEL HYPERPARAMETERS
cv_type = 'purged_kfold'
cv_number = 5


### IMPORT DATA
def import_data(data_path, remove_cols, contract='SPY'):
    # import data
    with pd.HDFStore(data_path + '/' + contract + '.h5') as store:
        data = store.get(contract)
    data.sort_index(inplace=True)
    
    # remove variables
    remove_cols = [col for col in remove_cols if col in data.columns]
    data.drop(columns=remove_cols, inplace=True)
    
    return data


remove_ohl = ['open', 'low', 'high', 'average', 'barCount',
              'open_vix', 'high_vix', 'low_vix', 'volume_vix']
data = import_data(DATA_PATH, remove_ohl, contract='SPY_raw')


### REGIME DEPENDENT ANALYSIS
if structural_break_regime == 'chow':
    if (data.loc[data['chow_segment'] == 1].shape[0] / 60 / 8) < 365:
        data = data.iloc[-(60*8*365):]
    else:
        data = data.loc[data['chow_segment'] == 1]
data = data.drop(columns=['chow_segment'])


### FILTER TRADING DAYS
daily_vol = ml.util.get_daily_vol(data['close'], lookback=50)
cusum_events = ml.filters.cusum_filter(data['close'], threshold=daily_vol.mean()*1)
data = data.reindex(cusum_events)


### CHOOSE LABELLING TECHNIQUE
X_cols = [col for col in data.columns if 'day_' not in col]
X = data[X_cols]
y_cols = [col for col in data.columns if label in col]
y_matrix = data[y_cols]
y_matrix.columns = ["t1",  "t_value", "ret", "bin"]

### REMOVE NA
remove_na_rows = y_matrix.isna().any(axis=1)
X = X.loc[~remove_na_rows]
y_matrix = y_matrix.loc[~remove_na_rows]
y_matrix.iloc[:, -1] = np.where(y_matrix.iloc[:, -1] == -1, 0, y_matrix.iloc[:, -1])


### REMOVE CORRELATED ASSETS
X = tml.modeling.preprocessing.remove_correlated_columns(
    data=X,
    columns_ignore=[],
    threshold=correlation_threshold)


### TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y_matrix.loc[:, y_matrix.columns.str.contains('bin')],
    test_size=0.10, shuffle=False, stratify=None)


### SAMPLE WEIGHTS
sample_weights_returns = ml.sample_weights.get_weights_by_return(
    y_matrix.reindex(X_train.index), X_train['close'], num_threads=1)
sample_weights_returns_time_decay = ml.sample_weights.get_weights_by_time_decay(
    y_matrix.reindex(X_train.index), X_train['close'], num_threads=1)


### DIMENSIONALITY REDUCTION
X_train_pca = pd.DataFrame(sklearn.preprocessing.scale(X_train), columns=X_train.columns)
X_test_pca = pd.DataFrame(sklearn.preprocessing.scale(X_test), columns=X_test.columns)
X_train_pca = pd.DataFrame(
    get_orthogonal_features(
        X_train_pca.drop(columns=['tick_rule'])),
    index=X_train_pca.index).add_prefix("PCA_")
pca_n_compenents = X_train_pca.shape[1]
X_test_pca = pd.DataFrame(
    get_orthogonal_features(
        X_test_pca.drop(columns=['tick_rule']),
        num_features=pca_n_compenents),
    index=X_test_pca.index).add_prefix("PCA_")


### SAVE FILES
file_names = ['X_train.pkl', 'y_train.pkl', 'X_test.pkl',
              'y_test.pkl', 'sample_weights_returns.pkl',
              'sample_weights_returns_time_decay.pkl', 'labeling_info.pkl']
tml.modeling.utils.save_files(
    [X_train, y_train, X_test, y_test, sample_weights_returns, sample_weights_returns_time_decay, y_matrix],
    file_names,
    os.getcwd())


### DELETE OLD FILES
def destroy_mfiles_object(mfiles_client, file_names):
    for f in file_names:
        try:
            search_result = mfiles_client.quick_search(f)
            object_id = search_result['Items'][0]['DisplayID']
            mfiles_client.destroy_object(object_type=0, object_id=int(object_id))
        except IndexError as ie:
            print(f'file {f} not in mfiles')


destroy_mfiles_object(mfiles_client, file_names=file_names)


### ADD FILES TO M-FILES
def save_to_mfiles(vault_id, file_names):
    # Connection details (replace as appropriate)
    MY_SERVER = "http://server.contentio.biz/REST/" # Enter your M-Files server address here
    MY_USER = "msagovac" # Enter your M-Files user name here
    MY_PASSWORD = "Wc8O10TaHz40" # Enter your M-Files password here
    MY_VAULT = vault_id # Enter your M-Files vault GUID here
    
    # File info for test file
    FILE_TYPE = "Dokument" # Replace with a object type defined in your server
    FILE_CLASS = "Dokument" # Replace with a object class defined in your server

    # Initialize MFilesClient and upload file
    my_client = mfiles.MFilesClient(server=MY_SERVER,
                                    user=MY_USER,
                                    password=MY_PASSWORD,
                                    vault=MY_VAULT)
    for f in file_names:
        # FILE_EXTRA_INFO = {
        #     "Name": FILE_NAME[:-4]
        # }
        my_client.upload_file(f, object_type=FILE_TYPE)


save_to_mfiles(vault_id="{452444F1-3175-43E5-BACF-5CD0159BFE97}", file_names=file_names)


# Download file
def read_mfiles(vault_id, file_names, path_to_save):
    # Connection details (replace as appropriate)
    MY_SERVER = "http://server.contentio.biz/REST/" # Enter your M-Files server address here
    MY_USER = "msagovac" # Enter your M-Files user name here
    MY_PASSWORD = "Wc8O10TaHz40" # Enter your M-Files password here
    MY_VAULT = vault_id # Enter your M-Files vault GUID here

    # Initialize MFilesClient and upload file
    my_client = mfiles.MFilesClient(server=MY_SERVER,
                                    user=MY_USER,
                                    password=MY_PASSWORD,
                                    vault=MY_VAULT)
    
    for f in file_names:
        my_client.download_file_name(f, local_path=path_to_save + f)
        


path_to_save = 'D:/ai_mfiles/'
file_names = ['X_train.pkl', 'y_train.pkl', 'X_test.pkl',
              'y_test.pkl', 'sample_weights_returns.pkl',
              'sample_weights_returns_time_decay.pkl', 'labeling_info.pkl']
mfiles_client = set_mfiles_client()
for f in file_names:
    mfiles_client.download_file_name(f, local_path=path_to_save + f)


read_mfiles(vault_id="{452444F1-3175-43E5-BACF-5CD0159BFE97}", file_names=file_names, path_to_save=path_to_save)
X_train = pd.read_pickle(Path(path_to_save + 'X_train.pkl'))
X_test = pd.read_pickle(Path(path_to_save + 'X_test.pkl'))
y_train = pd.read_pickle(Path(path_to_save + 'y_train.pkl'))
y_test = pd.read_pickle(Path(path_to_save + 'y_test.pkl'))
sample_weights_returns = pd.read_pickle(Path(path_to_save + 'sample_weights_returns.pkl'))
sample_weights_returns_time_decay = pd.read_pickle(Path(path_to_save + 'sample_weights_returns_time_decay.pkl'))
labeling_info = pd.read_pickle(Path(path_to_save + 'labeling_info.pkl'))


### MODELING

from sklearn.ensemble import RandomForestClassifier
from tune_sklearn import TuneSearchCV, TuneGridSearchCV


### CROS VALIDATION STEPS
if cv_type == 'purged_kfold':
    cv = ml.cross_validation.PurgedKFold(
        n_splits=cv_number,
        samples_info_sets=labeling_info['t1'].reindex(X_train.index))


### MODEL WITH SKLEARN

# estimator
rf = RandomForestClassifier(criterion='entropy',
                            class_weight='balanced_subsample')

# grid search
param_grid = {
    'max_depth': [2, 3, 4, 5],
    'n_estimators': [500, 1000],
    'max_features': [5, 10, 15, 20],
    'max_leaf_nodes': [4, 8, 16, 32]
    }
tune_search = TuneGridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    early_stopping=False,
    scoring='f1',
    n_jobs=16,
    cv=cv,
    verbose=1
)
tune_search.fit(X_train, y_train, sample_weight=labeling_info['t_value'].abs())
clf_predictions = tune_search.predict(X_test)
tune_search.cv_results_
tune_search.best_params_  #max_depth 3, n_estimators 1000, max_features 10, max_leaf_nodes 4


# random search
param_random = {
    "n_estimators": randint(50, 1000),
    "max_depth": randint(2, 3),
    'max_features': randint(5, 25),
    'min_weight_fraction_leaf': randint(0.03, 0.1)
}
tune_search = TuneGridSearchCV(
    estimator=rf,
    param_grid=param_random,
    early_stopping=False,
    n_iter=5,
    scoring='f1',
    n_jobs=16,
    cv=cv,
    verbose=1
)
tune_search.fit(X_train, y_train, sample_weight=sample_weigths)
clf_predictions = tune_search.predict(X_test)
tune_search.cv_results_
tune_search.best_index_
tune_search.best_params_


####### TRY TUNE-SKELARN WHEN THEY ANSWER ON MY ISSUES QUESTION #######
param_bayes = {
    "n_estimators": (50, 1000),
    "max_depth": (2, 7),
    'max_features': (1, 30)
    # 'min_weight_fraction_leaf': (0.03, 0.1, 'uniform')
}
tune_search = TuneSearchCV(
    rf,
    param_bayes,
    search_optimization='bayesian',
    max_iters=10,
    scoring='f1',
    n_jobs=16,
    cv=cv,
    verbose=1
)
tune_search.fit(X_train, y_train, sample_weight=sample_weigths)
clf_predictions = tune_search.predict(X_test)
tune_search.cv_results_
tune_search.best_index_
tune_search.best_params_
####### TRY TUNE-SKELARN WHEN THEY ANSWER ON MY ISSUES QUESTION #######


# clf = GridSearchCV(rf,
#                 param_grid=parameters,
#                 scoring='f1',
#                 n_jobs=16,
#                 cv=cv)
# clf.fit(X_train, y_train, sample_weight=sample_weigths)
# max_depth, n_features, max_leaf_nodes, n_estimators = clf.best_params_.values()

# model scores
# clf_predictions = clf.predict(X_test)
clf_f1_score = sklearn.metrics.f1_score(y_test, clf_predictions)
clf_accuracy_score = sklearn.metrics.accuracy_score(y_test, clf_predictions)
print(f'f1_score: {clf_f1_score}')
print(f'f1_score: {clf_f1_score}')
print(f'optimal_max_depth: {max_depth}')
print(f'optimal_n_features: {n_features}')
print(f'optimal_max_leaf_nodes {max_leaf_nodes}')
print(f'optimal_n_estimators {n_estimators}')
save_id = f'{max_depth}{n_features}{max_leaf_nodes}{n_estimators}{str(clf_f1_score)[2:6]}'
