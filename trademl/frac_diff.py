import glob
import os
import numpy as np
import pandas as pd
from numba import njit, prange
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mlfinlab.structural_breaks import (
    get_chu_stinchcombe_white_statistics,
    get_chow_type_stat, get_sadf)
import mlfinlab as ml
import mlfinlab.microstructural_features as micro
import trademl as tml
from trademl.modeling.utils import time_method


### HYPERPARAMETERS
save_path = 'D:/market_data/usa/ohlcv_features'
env_directory = None  # os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
median_outlier_thrteshold = 25


### IMPORT DATA
# import data from mysql database and 
contract = 'SPY'
q = 'SELECT date, open, high, low, close, volume FROM SPY'
data = tml.modeling.utils.query_to_db(q, 'odvjet12_market_data_usa')
data.set_index(data.date, inplace=True)
data.drop(columns=['date'], inplace=True)
data.sort_index(inplace=True)

# remove outliers
security = tml.modeling.outliers.remove_ourlier_diff_median(data, median_outlier_thrteshold)


### STATIONARITY
# save original ohlcv, I will need it later
ohlc = security[['open', 'high', 'low', 'close', 'volume']]

#  get dmin for every column
stationaryCols, min_d = tml.modeling.stationarity.min_ffd_all_cols(security)

# save to github for later 
min_dmin_d_save_for_backtesting = pd.Series(0, index=security.columns)
min_dmin_d_save_for_backtesting.update(min_d)
min_dmin_d_save_for_backtesting.dropna(inplace=True)
min_dmin_d_save_for_backtesting.to_csv(
    'C:/Users/Mislav/Documents/GitHub/trademl/data/min_d_' + contract + '.csv', sep=';')

# convert unstationary to stationary
security = tml.modeling.stationarity.unstat_cols_to_stat(security, min_d, stationaryCols)  # tml.modeling.stationarity.unstat_cols_to_stat
security = security.dropna()
security = security[stationaryCols].add_prefix('fracdiff_')

# merge orig_ohlc
security = security.merge(ohlc, how='left', left_index=True, right_index=True)


import glob
import os
import numpy as np
import pandas as pd
from numba import njit, prange
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mlfinlab.structural_breaks import (
    get_chu_stinchcombe_white_statistics,
    get_chow_type_stat, get_sadf)
import mlfinlab as ml
import mlfinlab.microstructural_features as micro
import trademl as tml
from trademl.modeling.utils import time_method


### HYPERPARAMETERS
save_path = 'D:/market_data/usa/ohlcv_features'
env_directory = None  # os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
median_outlier_thrteshold = 25


### IMPORT DATA
# import data from mysql database and 
contract = 'SPY'
q = 'SELECT date, open, high, low, close, volume FROM SPY'
data = tml.modeling.utils.query_to_db(q, 'odvjet12_market_data_usa')
data.set_index(data.date, inplace=True)
data.drop(columns=['date'], inplace=True)
data.sort_index(inplace=True)

# remove outliers
security = tml.modeling.outliers.remove_ourlier_diff_median(data, median_outlier_thrteshold)


### 1) STATIONARITY
# save original ohlcv, I will need it later
ohlc = security[['open', 'high', 'low', 'close', 'volume']]

#  get dmin for every column
stationaryCols, min_d = tml.modeling.stationarity.min_ffd_all_cols(security)

# save to github for later 
min_dmin_d_save_for_backtesting = pd.Series(0, index=security.columns)
min_dmin_d_save_for_backtesting.update(min_d)
min_dmin_d_save_for_backtesting.dropna(inplace=True)
min_dmin_d_save_for_backtesting.to_csv(
    'C:/Users/Mislav/Documents/GitHub/trademl/data/min_d_' + contract + '.csv', sep=';')

# convert unstationary to stationary
security = tml.modeling.stationarity.unstat_cols_to_stat(security, min_d, stationaryCols)  # tml.modeling.stationarity.unstat_cols_to_stat
security = security.dropna()
security = security[stationaryCols].add_prefix('fracdiff_')

# merge orig_ohlc
security = security.merge(ohlc, how='left', left_index=True, right_index=True)

# rearenge columns
security['date'] = security.index
security = security[['date', 'open', 'high', 'low', 'close', 'volume',
                     'fracdiff_open', 'fracdiff_high', 'fracdiff_low', 'fracdiff_close']]


### 2) STRUCTURAL BRAKES
# convert data to hourly to make code faster and decrease random component
close_hourly = security['fracdiff_close'].resample('H').last().dropna()
close_hourly = np.log(close_hourly)

# Chow-Type Dickey-Fuller Test
chow = tml.modeling.structural_breaks.get_chow_type_stat(
    series=close_hourly, min_length=10)
breakdate = chow.loc[chow == chow.max()]
security['chow_segment'] = 0
security['chow_segment'][breakdate.index[0]:] = 1
security['chow_segment'].loc[breakdate.index[0]:] = 1
security['chow_segment'] = np.where(security.index < breakdate.index[0], 0, 1)
security['chow_segment'].value_counts()


### SAVE
tml.modeling.utils.write_to_db(security, 'odvjet12_market_data_usa', contract + '_clean')
