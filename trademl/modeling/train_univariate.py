from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import pandas as pd
import numpy as mp
from pathlib import Path
import os


### TENSORBORADX WRITER
log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)


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


# CHOOSE FEATURE
column_where = np.where(col_names == 'open_vix')[0]
X_train = X_train[:1000, :, column_where]
y_train = y_train[:1000]
X_test = X_test[:1000, :, [column_where]]
y_test = y_test[:1000]


# KNeighbors Classifier
clf = KNeighborsTimeSeriesClassifier(n_neighbors=2,
                                     metric="dtw",
                                     n_jobs=8)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
predictions_proba = clf.predict_proba(X_test)
predictions_proba


### CLUSTERING
# KernelKMeans
from tslearn.clustering import KernelKMeans
from tslearn.generators import random_walks
X = random_walks(n_ts=50, sz=32, d=1)
X.shape
X_train.shape
gak_km = KernelKMeans(n_clusters=3, kernel="gak", random_state=0)
gak_km.fit(X_train)
gak_km.get_params()
X_train_classes = gak_km.predict(X_train)


# 

