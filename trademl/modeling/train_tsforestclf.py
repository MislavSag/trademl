import pandas as pd
import numpy as mp
from pathlib import Path
import os
import tslearn
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sktime.classification.compose import TimeSeriesForestClassifier



### TENSORBORADX WRITER
# log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
#     "%Y%m%d-%H%M%S")
# writer = SummaryWriter(log_dir)


### MODEL HYPERPARAMETERS
input_data_path = 'D:/algo_trading_files'
# model


### IMPORT PREPARED DATA
X_train = np.load(os.path.join(Path(input_data_path), 'X_train_seq.npy'))
X_test = np.load(os.path.join(Path(input_data_path), 'X_test_seq.npy'))
X_val = np.load(os.path.join(Path(input_data_path), 'X_val_seq.npy'))
y_train = np.load(os.path.join(Path(input_data_path), 'y_train_seq.npy'))
y_test = np.load(os.path.join(Path(input_data_path), 'y_test_seq.npy'))
y_val = np.load(os.path.join(Path(input_data_path), 'y_val_seq.npy'))
col_names = pd.read_csv(os.path.join(Path(input_data_path), 'col_names.csv'))