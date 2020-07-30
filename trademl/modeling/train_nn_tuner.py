# # fundamental modules
# import numpy as np
# import pandas as pd
# from numba import njit
# import matplotlib.pyplot as plt
# import matplotlib
# import joblib
# import json
# import sys
# import os
# # preprocessing
# import sklearn
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import kerastuner as kt
# # import tensorflow_addons as tfa
# # feature importance
# import shap
# from boruta import BorutaPy
# # finance packagesb
# import mlfinlab as ml
# import trademl as tml
# import IPython
# # import vectorbt as vbt


# # TENSORFLOW ATTRIBUTES
# print("Num GPUs Available: ", len(
#     tf.config.experimental.list_physical_devices('GPU')))

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# # tf.keras.backend.set_floatx('float64')  # see https://github.com/tensorflow/tensorflow/issues/41288


# ### DON'T SHOW GRAPH OPTION
# matplotlib.use("Agg")


# ### GLOBALS
# DATA_PATH = 'D:/market_data/usa/ohlcv_features/'


# ### IMPORT DATA
# contract = ['SPY']
# with pd.HDFStore(DATA_PATH + contract[0] + '.h5') as store:
#     data = store.get(contract[0])
# data.sort_index(inplace=True)


# ### CHOOSE/REMOVE VARIABLES
# remove_ohl = ['open', 'low', 'high', 'average', 'barCount',
#               'open_vix', 'high_vix', 'low_vix', 'close_vix', 'volume_vix',
#               'open_orig', 'high_orig', 'low_orig']
# remove_ohl = [col for col in remove_ohl if col in data.columns]
# data.drop(columns=remove_ohl, inplace=True)  #correlated with close


# ### NON-MODEL HYPERPARAMETERS
# num_threads = 1
# structural_break_regime = 'all'
# labeling_technique = 'trend_scanning'
# std_outlier = 10
# tb_volatility_lookback = 500
# tb_volatility_scaler = 1
# tb_triplebar_num_days = 10
# tb_triplebar_pt_sl = [1, 1]
# tb_triplebar_min_ret = 0.004
# ts_look_forward_window = 4800  # 60 * 8 * 10 (10 days)
# ts_min_sample_length = 30
# ts_step = 5
# tb_min_pct = 0.10
# sample_weights_type = 'returns'
# cv_type = 'purged_kfold'
# cv_number = 4
# rand_state = 3
# stationary_close_lables = False


# ### MODEL HYPERPARAMETERS



# ### USE STATIONARY CLOSE TO CALCULATE LABELS
# if stationary_close_lables:
#     data['close_orig'] = data['close']  # with original close reslts are pretty bad!


# ### REMOVE INDICATORS WITH HIGH PERIOD
# # if remove_ind_with_high_period:
# #     data.drop(columns=['DEMA96000', 'ADX96000', 'TEMA96000',
# #                        'ADXR96000', 'TRIX96000'], inplace=True)
# #     data.drop(columns=['autocorr_1', 'autocorr_2', 'autocorr_3',
# #                        'autocorr_4', 'autocorr_5'], inplace=True)
# #     print('pass')
    


# ### REGIME DEPENDENT ANALYSIS
# if structural_break_regime == 'chow':
#     if (data.loc[data['chow_segment'] == 1].shape[0] / 60 / 8) < 365:
#         data = data.iloc[-(60*8*365):]
#     else:
#         data = data.loc[data['chow_segment'] == 1]


# ### USE STATIONARY CLOSE TO CALCULATE LABELS
# if stationary_close_lables:
#     data['close_orig'] = data['close']  # with original close reslts are pretty bad!


# ### LABELING
# if labeling_technique == 'triple_barrier':
#     # TRIPLE BARRIER LABELING
#     triple_barrier_pipe= tml.modeling.pipelines.TripleBarierLabeling(
#         close_name='close_orig',
#         volatility_lookback=tb_volatility_lookback,
#         volatility_scaler=tb_volatility_scaler,
#         triplebar_num_days=tb_triplebar_num_days,
#         triplebar_pt_sl=tb_triplebar_pt_sl,
#         triplebar_min_ret=tb_triplebar_min_ret,
#         num_threads=1,
#         tb_min_pct=tb_min_pct
#     )
#     tb_fit = triple_barrier_pipe.fit(data)
#     labeling_info = tb_fit.triple_barrier_info
#     X = tb_fit.transform(data)
# elif labeling_technique == 'trend_scanning':
#     trend_scanning_pipe = tml.modeling.pipelines.TrendScanning(
#         close_name='close_orig',
#         volatility_lookback=tb_volatility_lookback,
#         volatility_scaler=tb_volatility_scaler,
#         ts_look_forward_window=ts_look_forward_window,
#         ts_min_sample_length=ts_min_sample_length,
#         ts_step=ts_step
#         )
#     labeling_info = trend_scanning_pipe.fit(data)
#     X = trend_scanning_pipe.transform(data)

# ###### TEST
# # X_TEST = X.iloc[:5000]
# # labeling_info_TEST = labeling_info.iloc[:5000]
# ###### TEST

# # TRAIN TEST SPLIT
# X_train, X_test, y_train, y_test = train_test_split(
#     X.drop(columns=['close_orig', 'chow_segment']), labeling_info['bin'],
#     test_size=0.10, shuffle=False, stratify=None)
    

# ### SAMPLE WEIGHTS (DECAY FACTOR CAN BE ADDED!)
# if sample_weights_type == 'returns':
#     sample_weigths = ml.sample_weights.get_weights_by_return(
#         labeling_info.reindex(X_train.index),
#         data.loc[X_train.index, 'close_orig'],
#         num_threads=1)
# elif sample_weights_type == 'time_decay':
#     sample_weigths = ml.sample_weights.get_weights_by_time_decay(
#         labeling_info.reindex(X_train.index),
#         data.loc[X_train.index, 'close_orig'],
#         decay=0.5, num_threads=1)
# elif labeling_technique is 'trend_scanning':
#     sample_weigths = labeling_info['t_value'].reindex(X_train.index).abs()


# ### PREPARE LSTM
# # X_train_lstm = .drop(columns=['close_orig']).values
# x = X_train.values
# y = y_train.values.reshape(-1, 1)
# # y = y.astype(str)
# x_test = X_test.values
# y_test_ = y_test.values.reshape(-1, 1)

# train_val_index_split = 0.75
# train_generator = keras.preprocessing.sequence.TimeseriesGenerator(
#     data=x,
#     targets=y,
#     length=100,
#     sampling_rate=1,
#     stride=1,
#     start_index=0,
#     end_index=int(train_val_index_split*x.shape[0]),
#     shuffle=False,
#     reverse=False,
#     batch_size=128
# )
# validation_generator = keras.preprocessing.sequence.TimeseriesGenerator(
#     data=x,
#     targets=y,
#     length=100,
#     sampling_rate=1,
#     stride=1,
#     start_index=int((train_val_index_split*x.shape[0] + 1)),
#     end_index=None,  #int(train_test_index_split*X.shape[0])
#     shuffle=False,
#     reverse=False,
#     batch_size=128
# )
# test_generator = keras.preprocessing.sequence.TimeseriesGenerator(
#     data=x_test,
#     targets=y_test_,
#     length=100,
#     sampling_rate=1,
#     stride=1,
#     start_index=0,
#     end_index=None,
#     shuffle=False,
#     reverse=False,
#     batch_size=128
# )


# # convert generator to inmemory 3D series (if enough RAM)
# def generator_to_obj(generator):
#     xlist = []
#     ylist = []
#     for i in range(len(generator)):
#         x, y = generator[i]
#         xlist.append(x)
#         ylist.append(y)
#     X_train = np.concatenate(xlist, axis=0)
#     y_train = np.concatenate(ylist, axis=0)
#     return X_train, y_train


# X_train_lstm, y_train_lstm = generator_to_obj(train_generator)
# X_val_lstm, y_val_lstm = generator_to_obj(validation_generator)
# X_test_lstm, y_test_lstm = generator_to_obj(test_generator)

# # test for shapes
# print('X and y shape train: ', X_train_lstm.shape, y_train_lstm.shape)
# print('X and y shape validate: ', X_val_lstm.shape, y_val_lstm.shape)
# print('X and y shape test: ', X_test_lstm.shape, y_test_lstm.shape)

# # change -1 to 1
# for i, y in enumerate(y_train_lstm):
#     if y == -1.:
#         y_train_lstm[i,:] = 0. 
# for i, y in enumerate(y_val_lstm):
#     if y == -1.:
#         y_val_lstm[i,:] = 0. 
# for i, y in enumerate(y_test_lstm):
#     if y == -1.:
#         y_test_lstm[i,:] = 0. 

# # change labels type to integer64
# y_train_lstm = y_train_lstm.astype(np.int64)
# y_val_lstm = y_val_lstm.astype(np.int64)
# y_test_lstm = y_test_lstm.astype(np.int64)
 

# ### MODEL


# def lstm_model(hp):
#     # define the modelwith hyperparameters
#     # GPU doesn't support recurrent droput: https://github.com/tensorflow/tensorflow/issues/40944
#     model = keras.Sequential()
#     hp_units = hp.Int('units', min_value = 32, max_value = 256, step = 32)
#     model.add(layers.LSTM(hp_units,
#                           return_sequences=True,
#                           input_shape=[None, x.shape[1]]))
#     model.add(layers.LSTM(hp_units))
#     model.add(layers.Dense(1, activation='sigmoid'))
    
#     # compile the model
#     hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 
#     model.compile(loss='binary_crossentropy',
#                   optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
#                   metrics=['accuracy']
#                   )
#                         #    keras.metrics.AUC(),
#                         #    keras.metrics.Precision(),
#                         #    keras.metrics.Recall()])
#     return model


# # model = keras.models.Sequential([
# #         keras.layers.LSTM(516, return_sequences=True, input_shape=[None, x.shape[1]]),
# #         keras.layers.LSTM(124, return_sequences=True, dropout=0.15),  # recurrent_dropout=0
# #         keras.layers.LSTM(32),  # recurrent_dropout=0
# #         keras.layers.Dense(1, activation='sigmoid',
# #                         kernel_initializer=keras.initializers.he_normal(seed=1))
        
# # ])

# # define tuner
# # tuner = kt.tuners.RandomSearch(lstm_model,
# #                                objective='val_accuracy',
# #                                max_trials=2,
# #                                executions_per_trial=2,
# #                                directory='lstm_tuner',
# #                                project_name='stock_prediction_lstm')
# tuner = kt.Hyperband(lstm_model,
#                      objective = 'val_accuracy', 
#                      max_epochs = 20,
#                      factor = 3,
#                      directory = 'my_dir',
#                      project_name = 'intro_to_kt')   
# tuner.search_space_summary()


# # callbacks
# early_stopping_cb = keras.callbacks.EarlyStopping(
#     patience=15, restore_best_weights=True)
# # newly defined callbacks
# class ClearTrainingOutput(tf.keras.callbacks.Callback):
#     def on_train_end(*args, **kwargs):
#         IPython.display.clear_output(wait = True)



# # fit tuner
# tuner.search(X_train_lstm,
#              y_train_lstm,
#              epochs=50,
#              batch_size=128,
#              shuffle=False,
#              validation_data=(X_val_lstm, y_val_lstm),
#              callbacks=[ClearTrainingOutput()]
# )

# # tuner results
# models = tuner.get_best_models(num_models=2)
# tuner.results_summary()


# # log message
# # print(f"""
# # The hyperparameter search is complete. The optimal number of units in the first densely-connected
# # layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
# # is {best_hps.get('learning_rate')}.
# # """)



# # get accuracy and score
# score, acc, auc, precision, recall = model.evaluate(
#     X_test_lstm, y_test_lstm, batch_size=128)
# print('score_train:', score)
# print('accuracy_train:', acc)
# print('auc_train:', auc)
# print('precision_train:', precision)
# print('recall_train:', recall)

# # get loss values and metrics
# historydf = pd.DataFrame(history.history)
# historydf.head(50)
 
# # predictions
# predictions = model.predict(X_test_lstm)
# predict_classes = model.predict_classes(X_test_lstm)

# # test metrics
# tml.modeling.metrics_summary.lstm_metrics(y_test_lstm, predict_classes)



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