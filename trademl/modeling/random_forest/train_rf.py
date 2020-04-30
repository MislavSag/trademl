# fundamental modules
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
from IPython.display import display, Image
# preprocessing
from sklearn.model_selection import train_test_split
# modelling
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier
from sklearn.base import clone
import xgboost
import shap
# metrics 
import mlfinlab as ml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    log_loss,
    )
from boruta import BorutaPy
# finance packagesb
import mlfinlab as ml
import trademl as tml


### IMPORT DATA
with pd.HDFStore(DATA_PATH) as store:
    spy = store.get('spy_with_vix')
spy.sort_index(inplace=True)
spy = spy.iloc[:50000]


### HYPER PARAMETERS
std_outlier = 10
tb_volatility_lookback = 50
tb_volatility_scaler = 1
tb_triplebar_num_days = 3
tb_triplebar_pt_sl = [1, 1]
tb_triplebar_min_ret = 0.003
sample_weights_type = 'returns'
cv_type = 'purged_kfold'
cv_number = 4


# def train():
#     data = mnist_data.load()
#     model = mnist_model.init()
#     model.train(data)

# if __name__ == "__main__":
#     train()