import os
from pathlib import Path
import pandas as pd
import trademl as tml
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from trademl.modeling.data_import import import_ohlcv
from trademl.modeling.outliers import RemoveOutlierDiffMedian
from trademl.modeling.features import AddFeatures
from trademl.modeling.stationarity import Fracdiff


# Parameters
save_path = 'D:/market_data/usa/ohlcv_features'
contract = 'SPY_IB'
keep_unstationary = True
frequency = 'M'

# Import data
data = import_ohlcv(save_path, contract=contract)

# Upsample
if frequency == 'H':
    data = data.resample('1H').agg({'open': 'first',
                                    'high': 'max', 
                                    'low': 'min', 
                                    'close': 'last',
                                    'volume': 'sum',
                                    'average': 'last',
                                    'barCount': 'sum'})
data = data.dropna()

# Preprocessing
pipe = make_pipeline(
    RemoveOutlierDiffMedian(median_outlier_thrteshold=25),
    AddFeatures(add_ta = False),
    Fracdiff(keep_unstationary=keep_unstationary)
    )
X = pipe.fit_transform(data)

# add radf from R
# if frequency == 'H':
#     radf = pd.read_csv(
#         'D:/algo_trading_files/exuber/radf_h_adf_4.csv',
#         sep=';', index_col=['Index'], parse_dates=['Index'])
#     radf = radf.resample('H').last()
#     X = pd.concat([X, radf[['radf']]], axis = 1).dropna()

# Save localy
save_path_local = os.path.join(Path(save_path), contract + '_clean' + '.h5')
if os.path.exists(save_path_local):
    os.remove(save_path_local)
X.to_hdf(save_path_local, contract + '_clean')
