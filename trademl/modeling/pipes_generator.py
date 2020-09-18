import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from trademl.modeling.structural_breaks import ChowStructuralBreakSubsample


# Parameters
input_data_path = 'D:/market_data/usa/ohlcv_features'
contract = 'SPY_IB'
chow_subsample = True
stationarity_method = None

# import data
file_name = contract + '_clean'
data = pd.read_hdf(os.path.join(Path(input_data_path), file_name + '.h5'), file_name)
data.sort_index(inplace=True)

# Choose subsamples and stationarity method
pipe = make_pipeline(
    ChowStructuralBreakSubsample(min_length=10) if chow_subsample else None,
    StationarityMethod(stationarity_method=None)
)
data = pipe.fit_transform(data)





### CHOOSE STATIONARY / UNSTATIONARY
if stationarity_tecnique == 'fracdiff':
    remove_cols = [col for col in data.columns if 'orig_' in col and col != 'orig_close']  
elif stationarity_tecnique == 'orig':
    remove_cols = [col for col in data.columns if 'fracdiff_' in col and col != 'orig_close']
data = data.drop(columns=remove_cols)




from sklearn.base import BaseEstimator, TransformerMixin
from trademl.modeling.utils import time_method





data_sample = data.iloc[:100000]
chow_pipe = StationarityMethod(stationarity_method=None)
X = chow_pipe.transform(data_sample)
X.head()



########## OVO NASTAVITI KASNIJE ##########

# import json
# import requests


# def find_min_d(series):
#     x = series.resample('D').last().dropna().values.tolist()
#     x = {'x': x}
#     x = json.dumps(x)
#     res = requests.post("http://46.101.219.193/plumber_test/mind", data=x)
#     res_json = res.json()[0]
#     return res_json


# def diffseries(series, min_d):
    
#     series = X['close'].iloc[:200000]
#     min_d = min_d_vector[0]
    
#     x = {'x': series.values.tolist(), 'min_d': min_d}
#     x = json.dumps(x)
#     res = requests.post("http://46.101.219.193/plumber_test/fracdiff", data=x)
#     res_json = res.json()
#     return res_json


# min_d_vector = X.apply(lambda x: find_min_d(x), axis=0)

# test = diffseries(X['close'], min_d_vector[0])

# for x in min_d_vector:



# add later
# sadf_linear =ml.structural_breaks.get_sadf(
#     close_weekly_log, min_length=20, add_const=True, model='linear', phi=0.5, num_threads=1, lags=5)

########## OVO NASTAVITI KASNIJE ##########


