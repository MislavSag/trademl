# fundamental modules
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import pyplot
from IPython.display import display, Image
from IPython.core.display import HTML 
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


def train():
    data = mnist_data.load()
    model = mnist_model.init()
    model.train(data)

if __name__ == "__main__":
    train()