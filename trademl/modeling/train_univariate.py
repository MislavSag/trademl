import pandas as pd
import numpy as np
from pathlib import Path
import os
import tslearn
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sktime.classification.compose import TimeSeriesForestClassifier
import sktime
from sklearn.metrics import (accuracy_score, confusion_matrix, recall_score,
                             precision_score, f1_score, classification_report,accuracy_score,
                             roc_curve)


### TENSORBORADX WRITER
# log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
#     "%Y%m%d-%H%M%S")
# writer = SummaryWriter(log_dir)


# MODEL HYPERPARAMETERS
input_data_path = 'D:/algo_trading_files'
# model


## IMPORT PREPARED DATA
X_train = np.load(os.path.join(Path(input_data_path), 'X_train_seq.npy'))
X_test = np.load(os.path.join(Path(input_data_path), 'X_test_seq.npy'))
X_val = np.load(os.path.join(Path(input_data_path), 'X_val_seq.npy'))
y_train = np.load(os.path.join(Path(input_data_path), 'y_train_seq.npy'))
y_test = np.load(os.path.join(Path(input_data_path), 'y_test_seq.npy'))
y_val = np.load(os.path.join(Path(input_data_path), 'y_val_seq.npy'))
col_names = pd.read_csv(os.path.join(Path(input_data_path), 'col_names.csv'))
col_names = col_names.iloc[:, 1]


# CONVERT TO SKTIME DATA TYPE
X_train = np.vstack([X_train, X_val])
X_train = tslearn.utils.to_sktime_dataset(X_train) 
X_test = tslearn.utils.to_sktime_dataset(X_test)
y_train = pd.Series(y_train.reshape(-1))
y_test = pd.Series(y_test.reshape(-1))

# test on smaller subset
sample_n = 100
X_train_sample = X_train.iloc[:sample_n, :]
y_train_sample = y_train[:sample_n]
X_test_sample = X_test.iloc[:sample_n, :]
y_test_sample = y_test[:sample_n]

for i, col in enumerate(col_names[:2]):
    print(col)

    # CHOOSE FEATURE
    X_train_step = X_train_sample.iloc[:, [i]]
    X_test_step = X_test_sample.iloc[:, [i]]

    ### TIME SERIES FOREST CLASSIFIER 
    classifier = TimeSeriesForestClassifier()
    classifier.fit(X_train_step, y_train_sample)
    y_pred = classifier.predict(X_test_step)
    
    ### METRICS
    print(f'accuracy_test: {accuracy_score(y_test_sample, y_pred)}')
    print(f"recall_test: {recall_score(y_test_sample, y_pred)}")
    print(f"precisoin_test: {precision_score(y_test_sample, y_pred)}")
    print(f"f1_test: {f1_score(y_test_sample, y_pred)}")




# # KNeighbors Classifier
# clf = KNeighborsTimeSeriesClassifier(n_neighbors=2,
#                                      metric="dtw",
#                                      n_jobs=8)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# predictions_proba = clf.predict_proba(X_test)
# predictions_proba


# ### CLUSTERING
# # KernelKMeans
# from tslearn.clustering import KernelKMeans
# from tslearn.generators import random_walks
# X = random_walks(n_ts=50, sz=32, d=1)
# X.shape
# X_train.shape
# gak_km = KernelKMeans(n_clusters=3, kernel="gak", random_state=0)
# gak_km.fit(X_train)
# gak_km.get_params()
# X_train_classes = gak_km.predict(X_train)

