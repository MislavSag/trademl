# import os
# from pathlib import Path
# import pandas as pd
# import numpy as np
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import train_test_split
# from trademl.modeling.structural_breaks import ChowStructuralBreakSubsample
# from trademl.modeling.stationarity import StationarityMethod
# from trademl.modeling.pipelines import TripleBarierLabeling


# # Parameters
# input_data_path = 'D:/market_data/usa/ohlcv_features'
# contract = 'SPY_IB'
# chow_subsample = False
# stationarity_method = None
# # labeling
# labeling = 'ts' # ts is trend_scanning / tb is tripple-barrier
# tb_triplebar_num_days = 10
# tb_triplebar_pt_sl = [1, 1]
# tb_triplebar_min_ret = 0.004
# ts_look_forward_window = 1200  # 60 * 8 * 10 (10 days)
# ts_min_sample_length = 30
# ts_step = 5
# tb_min_pct = 0.10

# # import data
# file_name = contract + '_clean'
# data = pd.read_hdf(os.path.join(Path(input_data_path), file_name + '.h5'), file_name)
# data.sort_index(inplace=True)

# # Choose subsamples, stationarity method and make labels
# pipe = make_pipeline(
#     ChowStructuralBreakSubsample(min_length=10) if chow_subsample else None,
#     StationarityMethod(stationarity_method=None),
#     TripleBarierLabeling(
#         tb_volatility_lookback, tb_volatility_scaler,
#         tb_triplebar_num_days, tb_triplebar_pt_sl,
#         tb_triplebar_min_ret, num_threads,
#         tb_min_pct) if labeling == 'tb' else None
#     TrendScanning(
#         tb_volatility_lookback, tb_volatility_scaler,
#         ts_look_forward_window, ts_min_sample_length,
#         ts_step) if labeling == 'ts' else None
# )
# data = pipe.fit_transform(data)

# # Categorical features
# categorial_features = ['tick_rule', 'HT_TRENDMODE', 'volume_vix']
# categorial_features = [col for col in categorial_features if col in data.columns]
# data = data.drop(columns=categorial_features)  # remove for now

# # train test split
# train, test = train_test_split(data, test_size=0.10, shuffle=False, stratify=None)


# from sklearn.base import BaseEstimator, TransformerMixin
# from trademl.modeling.utils import time_method





# data_sample = data.iloc[:100000]
# chow_pipe = StationarityMethod(stationarity_method=None)
# X = chow_pipe.transform(data_sample)
# X.head()



# ########## OVO NASTAVITI KASNIJE ##########

# # import json
# # import requests


# # def find_min_d(series):
# #     x = series.resample('D').last().dropna().values.tolist()
# #     x = {'x': x}
# #     x = json.dumps(x)
# #     res = requests.post("http://46.101.219.193/plumber_test/mind", data=x)
# #     res_json = res.json()[0]
# #     return res_json


# # def diffseries(series, min_d):
    
# #     series = X['close'].iloc[:200000]
# #     min_d = min_d_vector[0]
    
# #     x = {'x': series.values.tolist(), 'min_d': min_d}
# #     x = json.dumps(x)
# #     res = requests.post("http://46.101.219.193/plumber_test/fracdiff", data=x)
# #     res_json = res.json()
# #     return res_json


# # min_d_vector = X.apply(lambda x: find_min_d(x), axis=0)

# # test = diffseries(X['close'], min_d_vector[0])

# # for x in min_d_vector:



# # add later
# # sadf_linear =ml.structural_breaks.get_sadf(
# #     close_weekly_log, min_length=20, add_const=True, model='linear', phi=0.5, num_threads=1, lags=5)

# ########## OVO NASTAVITI KASNIJE ##########


