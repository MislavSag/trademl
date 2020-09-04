from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
import json
import sys
import os
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
import mlfinlab as ml
import trademl as tml
from tensorboardX import SummaryWriter
matplotlib.use("Agg")


### TENSORFLOW ATTRIBUTES
assert tf.config.list_physical_devices('GPU')
assert tf.config.list_physical_devices('GPU')[0][1] == 'GPU'
assert tf.test.is_built_with_cuda()


### TENSORBORADX WRITER
log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)


### MODEL HYPERPARAMETERS
input_data_path = 'D:/algo_trading_files'
use_pca_features = False
# model
train_val_index_split = 0.75
time_step_length = 120
batch_size = 128
n_lstm_layers = 3
n_units = 64
dropout = 0.2
lr = 10e-2
epochs = 50
optimizer = 'random'
max_trials = 10  # parameter for random optimizer
executions_per_trial = 2  # parameter for random optimizer


### IMPORT PREPARED DATA
if use_pca_features:
    X_train = np.load(os.path.join(Path(input_data_path), 'X_train_seq_pca.npy'))
    X_test = np.load(os.path.join(Path(input_data_path), 'X_test_seq_pca.npy'))
    X_val = np.load(os.path.join(Path(input_data_path), 'X_val_seq_pca.npy'))
    y_train = np.load(os.path.join(Path(input_data_path), 'y_train_seq_pca.npy'))
    y_test = np.load(os.path.join(Path(input_data_path), 'y_test_seq_pca.npy'))
    y_val = np.load(os.path.join(Path(input_data_path), 'y_val_seq_pca.npy'))
else:
    X_train = np.load(os.path.join(Path(input_data_path), 'X_train_seq.npy'))
    X_test = np.load(os.path.join(Path(input_data_path), 'X_test_seq.npy'))
    X_val = np.load(os.path.join(Path(input_data_path), 'X_val_seq.npy'))
    y_train = np.load(os.path.join(Path(input_data_path), 'y_train_seq.npy'))
    y_test = np.load(os.path.join(Path(input_data_path), 'y_test_seq.npy'))
    y_val = np.load(os.path.join(Path(input_data_path), 'y_val_seq.npy'))


### TEST ###
# X_train = X_train[:1000]
# y_train = y_train[:1000]
### TEST ###


### MODEL
def lstm_model(hp):
    # GPU doesn't support recurrent droput: https://github.com/tensorflow/tensorflow/issues/40944
    model = keras.Sequential()
    hp_units = hp.Int('units', min_value=32, max_value=256, step=32)
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, default=0.25, step=0.05)
    hp_num_layers = hp.Int('num_layers', 1, 3) 
    if hp_num_layers == 1:
        model.add(layers.LSTM(hp_units,
                              input_shape=[None, X_train.shape[2]]))
    elif hp_num_layers == 2:
        model.add(layers.LSTM(hp_units,
                              return_sequences=True,
                              input_shape=[None, X_train.shape[2]]))
        model.add(layers.LSTM(hp_units, dropout=hp_dropout))
    elif hp_num_layers == 3:
        model.add(layers.LSTM(hp_units,
                              return_sequences=True,
                              input_shape=[None, X_train.shape[2]]))
        model.add(layers.LSTM(hp_units, return_sequences=True, dropout=hp_dropout))
        model.add(layers.LSTM(hp_units, dropout=hp_dropout))
    model.add(layers.Dense(1, activation='sigmoid'))

    # compile the model
    hp_learning_rate = hp.Float('learning_rate',
                                min_value=1e-4,
                                max_value=1e-2,
                                sampling='LOG',
                                default=1e-3
                                ) 
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=['accuracy',
                           keras.metrics.AUC(),
                           keras.metrics.Precision(),
                           keras.metrics.Recall()]
                  )
    return model


# define tuner
#################### FOR TEST ##############################
optimizer = 'random'
#################### FOR TEST ##############################

if optimizer == 'random':
    tuner = kt.tuners.RandomSearch(lstm_model,
                                   objective='val_accuracy',
                                   max_trials=max_trials,
                                   executions_per_trial=executions_per_trial,
                                   directory='lstm_tuner',
                                   project_name='stock_prediction_lstm')
    tuner.search(X_train,
                 y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 shuffle=False,
                 validation_data=(X_val, y_val),
                 callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy', patience=5, restore_best_weights=True)]
    )
elif optimizer == 'hyperband':
    tuner = kt.Hyperband(lstm_model,
                        objective = 'val_accuracy', 
                        max_epochs = 30,
                        factor = 4,
                        directory = 'my_dir',
                        project_name = 'intro_to_kt')
    tuner.search(X_train,
                 y_train,
                 batch_size=batch_size,
                 shuffle=False,
                 validation_data=(X_val, y_val)
                 )

# Build the model with the optimal hyperparameters and train it on the data
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    X_train, y_train, epochs = epochs, validation_data = (X_val, y_val),
    callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy', patience=5, restore_best_weights=True)])

# save best model params
print('units: ', best_hps.get('units'))
print('dropout: ', best_hps.get('dropout'))
print('num_layers: ', best_hps.get('num_layers'))
print('learning_rate: ', best_hps.get('learning_rate'))

# get accuracy and score
score, acc, auc, precision, recall = model.evaluate(
    X_test, y_test, batch_size=batch_size)
print('score_validation:', score)
print('accuracy_validation:', acc)
print('auc_validation:', auc)
print('precision_validation:', precision)
print('recall_validation:', recall)

# get loss values and metrics
# historydf = pd.DataFrame(history.history)
# historydf.head(50)
 
# predictions
predictions = model.predict(X_test)
predict_classes = model.predict_classes(X_test)

# test metrics
tml.modeling.metrics_summary.lstm_metrics(y_test, predict_classes)



# ### SAVE MODELS AND FEATURES
# # save features names
# pd.Series(X_train.columns).to_csv('lstm_features.csv', sep=',')
# # save model
# with open('Output_Model.json', 'w') as json_file:
#     json_file.write(model.to_json())
# # save weigths to json
# weights = model.get_weights()
# import json
# class EncodeNumpyArray(json.JSONEncoder):
#     """
#     Encodes Numpy Array as JSON.
#     """
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)
# serialized_weights = json.dumps(weights, cls=EncodeNumpyArray)
# with open('Output_Weights.txt', 'w') as fh:
#     fh.write(serialized_weights)

# ## In QC
# # weigths
# with open('Output_Weights.txt') as text_file:
#     serialized_weights = json.load(text_file)
# weights = [np.array(w) for w in serialized_weights]
# # model
# with open('Output_Model.json') as json_file:
#     model_loaded = json_file.read()
# model_loaded = keras.models.model_from_json(model_loaded)
# model_loaded = model_loaded.set_weights(weights)
# # test
# predictions = model.predict(X_test_lstm)
# predict_classes = model.predict_classes(X_test_lstm)







# ### FEATURE IMPORTANCE

# # SHAP model explainer
# # explainer = shap.DeepExplainer(model, X_train_lstm)
# # shap_value = explainer.shap_values(X_test_lstm)
# # shap_val = np.array(shap_value)
# # a = np.absolute(shap_val[0])
# # b = np.sum(a, axis=1)
# # SHAP_list = [np.sum(b[:, 0]), np.sum(b[:, 1]), np.sum(b[:, 2]), np.sum(b[:, 3]), np.sum(b[:, 4])]
# # N_weight = normalize(weight_list)
# # N_SHAP = normalize(SHAP_list)


# # # save the model
# model.save('C:/Users/Mislav/Documents/GitHub/trademl/trademl/modeling/lstm_clf_cloud.h5')
# # model = keras.models.load_model('C:/Users/Mislav/Documents/GitHub/trademl/trademl/modeling/lstm_clf_cloud.h5')
# # predictions = model.predict(X_TEST)
# print("Saved model to disk")





# ### SAVE THE MODEL FOR REMOTE PREDICTIONS ###
# model_version = "0001"
# model_name = "lstm_cloud"
# model_path = os.path.join(model_name, model_version)
# tf.saved_model.save(model, model_path)
# model_path_full_path = 'C:/Users/Mislav/Documents/GitHub/trademl/trademl/modeling/' + model_path

# # test dataset
# X_TEST = X_test_lstm[[0], :, :]
# X_TEST

# # 1) load saved model and predict
# saved_model = tf.saved_model.load(model_path_full_path)
# y_pred = saved_model(X_TEST, training=False)
# ### SAVE THE MODEL FOR REMOTE PREDICTIONS ###





# # input_data_json = json.dumps({
# #     "signature_name": "lstm_spy",
# #     "instances": X_test_lstm.tolist(),
# # })

# # send post requests
# # import requests
# # SERVER_URL = 'http://localhost:8501/v1/models/my_mnist_model:predict'
# # response = requests.post(SERVER_URL, data=input_data_json)
# # response.raise_for_status() # raise an exception in case of error
# # response = response.json()


# # y_proba = np.array(response["predictions"])

# # # GRPC set up
# # 

# # from tensorflow_serving.apis.predict_pb2 import PredictRequest
# # request = PredictRequest()
# # request.model_spec.name = model_name
# # request.model_spec.signature_name = "serving_default"
# # input_name = model.input_names[0]
# # request.inputs[input_name].CopyFrom(tf.make_tensor_proto(X_TEST))  # X_test_lstm

# # # GRPC query
# # import grpc
# # from tensorflow_serving.apis import prediction_service_pb2_grpc

# # channel = grpc.insecure_channel('localhost:8500')
# # predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
# # response = predict_service.Predict(request, timeout=20.0)

# # output_name = model.output_names[0]
# # outputs_proto = response.outputs[output_name]
# # y_proba = tf.make_ndarray(outputs_proto)

# # type(X_test_lstm)
# # X_test_lstm.shape
# # model.predict(X_test_lstm[[0], :, :])



# # GOOGLE API CLIENT LIBRARY

# # key_path = 'C:/Users/Mislav/Downloads/mltrading-282913-c15767742784.json'
# # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

# # import googleapiclient.discovery
# # project_id = "mltrading-282913" # change this to your project ID
# # model_id = "lstm_trade_model"
# # model_path = "projects/{}/models/{}".format(project_id, model_id)
# # model_path = model_path + '/versions/0001'
# # ml_resource = googleapiclient.discovery.build("ml", "v1", cache_discovery=False).projects()


# # def predict(X):
# #     input_data_json = {"signature_name": "serving_default",
# #                        "instances": X.tolist()}
# #     request = ml_resource.predict(name=model_path, body=input_data_json)
# #     response = request.execute()
# #     if "error" in response:
# #         raise RuntimeError(response["error"])
# #     return np.array([pred[output_name] for pred in response["predictions"]])

# # Y_probas = predict(X_TEST)
# # np.round(Y_probas, 2)


# # input_data_json = {"signature_name": "serving_default",
# #                     "instances": X_TEST.tolist()}
# # request = ml_resource.predict(name=model_path, body=input_data_json)
# # response = request.execute()



# # #### github question

# # # import
# # import numpy as np
# # import json

# # # download model
# # file_url = 'https://github.com/MislavSag/trademl/blob/master/trademl/modeling/lstm_clf_cloud.h5?raw=true'
# # file_save_path = 'C:/Users/Mislav/Downloads/my_model.h5'  # CHANGE THIS PATH
# # tf.keras.utils.get_file(file_save_path, file_url)

# # # data

# # # 
# # model = keras.models.load_model(file_save_path)
# # predictions = model.predict(X_TEST)


# # from tensorflow.keras.datasets import imdb
# # (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=1000)

# # type(y_train)
# # y_train.dtype
