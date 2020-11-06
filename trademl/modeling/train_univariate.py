import pandas as pd
import numpy as np
from pathlib import Path
import os
import tslearn
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sktime.classification.compose import TimeSeriesForestClassifier
import sktime
import joblib
from sklearn.metrics import (accuracy_score, confusion_matrix, recall_score,
                             precision_score, f1_score, classification_report,accuracy_score,
                             roc_curve) ``



# Tensorboardx writer
# log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
#     "%Y%m%d-%H%M%S")
# writer = SummaryWriter(log_dir)


# Import data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
y_val = np.load('y_val.npy')
col_names = pd.read_csv('col_names.csv')
col_names = col_names.iloc[:, 1]


# Convert to sktime data type
X_train = np.vstack([X_train, X_val])
X_train = tslearn.utils.to_sktime_dataset(X_train) 
X_test = tslearn.utils.to_sktime_dataset(X_test)
y_train = np.vstack([y_train, y_val])
y_train = pd.Series(y_train.reshape(-1))
y_test = pd.Series(y_test.reshape(-1))


# Timeseries random foreset for every column
for i, col in enumerate(col_names[:2]):
    print(col)

    # Choose one feature
    X_train_step = X_train.iloc[:, [i]]
    X_test_step = X_test.iloc[:, [i]]

    # Time series forest clf
    classifier = TimeSeriesForestClassifier()
    classifier.fit(X_train_step, y_train)
    y_pred = classifier.predict(X_test_step)
    
    # Metrics
    print(f'accuracy_test: {accuracy_score(y_test, y_pred)}')
    print(f"recall_test: {recall_score(y_test, y_pred)}")
    print(f"precisoin_test: {precision_score(y_test, y_pred)}")
    print(f"f1_test: {f1_score(y_test, y_pred)}")



# clf2 = pickle.loads(s)
# clf2.predict(X_test[0:1])


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


