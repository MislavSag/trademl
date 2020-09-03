import pandas as pd
import os
from pathlib import Path
import json
import h2o
import numpy as np
from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree._tree import Tree
from sklearn.ensemble import RandomForestClassifier
import time
from functools import wraps
from sqlalchemy import create_engine
import mfiles
from os import environ, path
from dotenv import load_dotenv



def cbind_pandas_h2o(X_train, y_train):
    """
    Convert padnas df to h2o df and cbind X and y.

    :param X_train: (pd.DataFrame) pandas data frame X
    :param y_train: (pd.Series) pandas Sereis <_train
    :return: (h2o.frame.H2Oframe) merged X and y h2o df
    """
    X_train_h2o = h2o.H2OFrame(X_train)
    y_train_h2o = h2o.H2OFrame(y_train.to_frame())
    y_train_h2o['bin'] = y_train_h2o['bin'].asfactor()
    train = X_train_h2o.cbind(y_train_h2o)
    return train


def serialize_tree(tree):
    serialized_tree = tree.__getstate__()

    dtypes = serialized_tree['nodes'].dtype
    serialized_tree['nodes'] = serialized_tree['nodes'].tolist()
    serialized_tree['values'] = serialized_tree['values'].tolist()

    return serialized_tree, dtypes

# def deserialize_tree(tree_dict, n_features, n_classes, n_outputs):
#     tree_dict['nodes'] = [tuple(lst) for lst in tree_dict['nodes']]

#     names = ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples']
#     tree_dict['nodes'] = np.array(tree_dict['nodes'], dtype=np.dtype({'names': names, 'formats': tree_dict['nodes_dtype']}))
#     tree_dict['values'] = np.array(tree_dict['values'])

#     tree = sklearn.tree._tree.Tree(n_features, np.array([n_classes], dtype=np.intp), n_outputs)
#     tree.__setstate__(tree_dict)

#     return tree

def serialize_decision_tree(model):
    tree, dtypes = serialize_tree(model.tree_)
    serialized_model = {
        'meta': 'decision-tree',
        'feature_importances_': model.feature_importances_.tolist(),
        'max_features_': model.max_features_,
        'n_classes_': int(model.n_classes_),
        'n_features_': model.n_features_,
        'n_outputs_': model.n_outputs_,
        'tree_': tree,
        'classes_': model.classes_.tolist(),
        'params': model.get_params()
    }


    tree_dtypes = []
    for i in range(0, len(dtypes)):
        tree_dtypes.append(dtypes[i].str)

    serialized_model['tree_']['nodes_dtype'] = tree_dtypes

    return serialized_model

# def deserialize_decision_tree(model_dict):
#     deserialized_model = DecisionTreeClassifier(**model_dict['params'])   

#     deserialized_model.classes_ = np.array(model_dict['classes_'])
#     deserialized_model.max_features_ = model_dict['max_features_']
#     deserialized_model.n_classes_ = model_dict['n_classes_']
#     deserialized_model.n_features_ = model_dict['n_features_']
#     deserialized_model.n_outputs_ = model_dict['n_outputs_']

#     tree = deserialize_tree(model_dict['tree_'], model_dict['n_features_'], model_dict['n_classes_'], model_dict['n_outputs_'])
#     deserialized_model.tree_ = tree

#     return deserialized_model

def serialize_random_forest(model):
    serialized_model = {
        'meta': 'rf',
        'max_depth': model.max_depth,
        'min_samples_split': model.min_samples_split,
        'min_samples_leaf': model.min_samples_leaf,
        'min_weight_fraction_leaf': model.min_weight_fraction_leaf,
        'max_features': model.max_features,
        'max_leaf_nodes': model.max_leaf_nodes,
        'min_impurity_decrease': model.min_impurity_decrease,
        'min_impurity_split': model.min_impurity_split,
        'n_features_': model.n_features_,
        'n_outputs_': model.n_outputs_,
        'classes_': model.classes_.tolist(),
        'estimators_': [serialize_decision_tree(decision_tree) for decision_tree in model.estimators_],
        'params': model.get_params()
    }

    if 'oob_score_' in model.__dict__:
        serialized_model['oob_score_'] = model.oob_score_
    if 'oob_decision_function_' in model.__dict__:
        serialized_model['oob_decision_function_'] = model.oob_decision_function_.tolist()

    if isinstance(model.n_classes_, int):
        serialized_model['n_classes_'] = model.n_classes_
    else:
        serialized_model['n_classes_'] = model.n_classes_.tolist()

    return serialized_model

def deserialize_random_forest(model_dict):
    model = RandomForestClassifier(**model_dict['params'])
    estimators = [deserialize_decision_tree(decision_tree) for decision_tree in model_dict['estimators_']]
    model.estimators_ = np.array(estimators)

    model.classes_ = np.array(model_dict['classes_'])
    model.n_features_ = model_dict['n_features_']
    model.n_outputs_ = model_dict['n_outputs_']
    model.max_depth = model_dict['max_depth']
    model.min_samples_split = model_dict['min_samples_split']
    model.min_samples_leaf = model_dict['min_samples_leaf']
    model.min_weight_fraction_leaf = model_dict['min_weight_fraction_leaf']
    model.max_features = model_dict['max_features']
    model.max_leaf_nodes = model_dict['max_leaf_nodes']
    model.min_impurity_decrease = model_dict['min_impurity_decrease']
    model.min_impurity_split = model_dict['min_impurity_split']

    if 'oob_score_' in model_dict:
        model.oob_score_ = model_dict['oob_score_']
    if 'oob_decision_function_' in model_dict:
        model.oob_decision_function_ = model_dict['oob_decision_function_']

    if isinstance(model_dict['n_classes_'], list):
        model.n_classes_ = np.array(model_dict['n_classes_'])
    else:
        model.n_classes_ = model_dict['n_classes_']

    return model


def load_model(url):
    model = deserialize_random_forest(json.loads(url))
    return model


def time_method(func):
    @wraps(func)
    def timed(*args, **kw):
        time_thresh = 1 # Function time taken printed if greater than this number
        
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()
        
        if te - ts > time_thresh:
            print("%r took %2.2f seconds to run." % (func.__name__, te - ts))

        return result

    return timed


def write_to_db(df, database_name, table_name, primary_key=True):
    """
    Creates a sqlalchemy engine and write the dataframe to database
    Source: https://stackoverflow.com/questions/55750229/how-to-save-a-data-frame-as-a-table-in-sql
    """
    # replacing infinity by nan
    df = df.replace([np.inf, -np.inf], np.nan)

    # create sqlalchemy engine
    engine = create_engine("mysql+pymysql://{user}:{pw}@91.234.46.219/{db}"
                           .format(user="odvjet12_mislav",
                                   pw="Theanswer0207",
                                   db=database_name))

    # Write to DB
    df.to_sql(table_name, engine, if_exists='replace', index=False, chunksize=100)
    with engine.connect() as con:
        con.execute('ALTER table ' + table_name + ' add id int primary key auto_increment;')
        
        
def query_to_db(query, database_name):
    """
    Get dat from database.
    """
    # create sqlalchemy engine
    engine = create_engine("mysql+pymysql://{user}:{pw}@91.234.46.219/{db}"
                           .format(user="odvjet12_mislav",
                                   pw="Theanswer0207",
                                   db=database_name))

    # Write to DB
    return pd.read_sql(query, engine)



def write_to_db_update(df, database_name, table_name):
    """
    Creates a sqlalchemy engine and write the dataframe to database
    Source: https://stackoverflow.com/questions/55750229/how-to-save-a-data-frame-as-a-table-in-sql
    """
    # replacing infinity by nan
    df = df.replace([np.inf, -np.inf], np.nan)

    # create sqlalchemy engine
    engine = create_engine("mysql+pymysql://{user}:{pw}@91.234.46.219/{db}"
                           .format(user="odvjet12_mislav",
                                   pw="Theanswer0207",
                                   db=database_name))

    # Write to DB
    df.to_sql(table_name, engine, if_exists='append', index=False, chunksize=100)


def balance_multiclass(series, grid=np.arange(1, 10, 0.1)):
    ideal_ratio = [0.33, 0.33, 0.33]
    devergence = []
    for i in range(len(grid)):
        grid_bin = np.where((series > grid[i]).values, 1, 0)
        grid_bin = np.where((series < -grid[i]).values, -1, grid_bin)
        tbl_freq = pd.Series(grid_bin).value_counts() / grid_bin.shape[0]
        devergence.append((tbl_freq - ideal_ratio).max())
    optimal_threshold = grid[np.array(devergence).argmin()]
    balanced_bins = np.where((series > optimal_threshold).values, 1, 0)
    balanced_bins = np.where((series < -optimal_threshold).values, -1, balanced_bins)
    
    return balanced_bins


def save_files(objects, file_names, directory='important_features'):
    """
    Save file to specific deirectory.    
    params
    """
    # create directory if it does not exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # save files to directory
    for df, file_name in zip(objects, file_names):
        saving_path = Path(f'{directory}/{file_name}')
        if ".csv" in file_name: 
            df.to_csv(saving_path)
        elif ".pkl" in file_name:
            df.to_pickle(saving_path)
        elif "." not in file_name:
            np.save(saving_path, df)


def set_mfiles_client(env_directory):
    """
    Set up mfiles client
    """
    ### GET M-FILES CREDITENTIPALS
    load_dotenv(env_directory)
    SERVER = os.environ.get('SERVER')
    USER = os.environ.get('USER')
    PASSWORD = os.environ.get('PASSWORD')
    VAULT = os.environ.get('VAULT')
    mfiles_client = mfiles.MFilesClient(server=SERVER,
                                        user=USER,
                                        password=PASSWORD,
                                        vault=VAULT)
    return mfiles_client


def destroy_mfiles_object(mfiles_client, file_names):
    if isinstance(file_names, list):
        for f in file_names:
            search_result = mfiles_client.quick_search(f)
            if search_result['Items'] == []:
                print(f'file {f} not in mfiles')
            else:
                object_id = search_result['Items'][0]['DisplayID']
                mfiles_client.destroy_object(object_type=0, object_id=int(object_id))
    else:
        raise('file_names argument must be a list')
