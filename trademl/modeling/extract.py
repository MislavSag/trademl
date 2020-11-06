import re
import datetime
import os
from pathlib import Path
import pandas as pd
import trademl as tml
import ib_insync
from ib_insync import *
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from trademl.modeling.data_import import import_ohlcv
from trademl.modeling.outliers import RemoveOutlierDiffMedian
from trademl.modeling.features import AddFeatures
from trademl.modeling.stationarity import Fracdiff
util.startLoop()  # uncomment this line when in a notebook


# Parameters
save_path = 'D:/market_data/usa/ohlcv_features'
contract = 'SPY_IB'
keep_unstationary = True
frequency = '1 hour'  # pandas freq
tickers = ['SPY', 'AAPL', 'AMZN', 'T', 'UAL']

# Test if frequency is in IB frequency
ib_freq = ['1 secs', '5 secs', '10 secs' '15 secs',
           '30 secs', '1 min', '2 mins', '3 mins',
           '5 mins', '10 mins', '15 mins', '20 mins',
           '30 mins', '1 hour', '2 hours', '3 hours',
           '4 hours', '8 hours', '1 day', '1 week', '1 month']
if frequency not in ib_freq:
    print('Choose frequency compatible with IB.')
    raise
    
# IB connection
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=2)

# Get data from IB
if re.search(r'hour|day|week|month', frequency):
    dfs = []
    for tick in tickers:
        con = Stock(tick, 'SMART', 'USD')
        print(con)
        bars = ib.reqHistoricalData(
            contract=con,
            endDateTime=datetime.datetime.now(),
            durationStr='20 Y',
            barSizeSetting=frequency,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1,
            timeout=60*10
        )
        df = util.df(bars)
        multiindex = [[tick], df.date.tolist()]
        multiindex = pd.MultiIndex.from_product(
            multiindex, names=['ticker', 'time'])
        df.index = multiindex
        df = df.drop(columns=['date'])
        dfs.append(df)

    # merge
    market_data = pd.concat(dfs, axis=0)

    # disconnect interactive brokers
    ib.disconnect()


############## Old import function
# data = import_ohlcv(save_path, contract=contract)

# Upsample
# if frequency:
#     data = data.resample(frequency).agg({'open': 'first',
#                                          'high': 'max',
#                                          'low': 'min',
#                                          'close': 'last',
#                                          'volume': 'sum',
#                                          'average': 'last',
#                                          'barCount': 'sum'})
# data = data.dropna()
############## Old import function

# Preprocessing
pipe = make_pipeline(
    RemoveOutlierDiffMedian(median_outlier_thrteshold=25),
    # AddFeatures(add_ta = False),
    Fracdiff(keep_unstationary=keep_unstationary)
    )
X = pipe.fit_transform(market_data.loc[['SPY']].droplevel(0))


min_ffd_all_cols(market_data.loc[['T']].droplevel(0))

for date, new_df in market_data.groupby(level=0):
    print(new_df)

test = (market_data.groupby(level=0)
        .apply(lambda x: pipe.fit_transform(x.droplevel(level=0))))
market_data.groupby(level=0).mean()

# add radf from R
# if frequency == 'H':
#     radf = pd.read_csv(
#         'D:/algo_trading_files/exuber/radf_h_adf_4.csv',
#         sep=';', index_col=['Index'], parse_dates=['Index'])
#     radf = radf.resample('H').last()
#     X = pd.concat([X, radf[['radf']]], axis = 1).dropna()

# Save localy
save_path_local = os.path.join(
    Path(save_path),
    contract + '_clean' + '.h5')
if os.path.exists(save_path_local):
    os.remove(save_path_local)
X.to_hdf(save_path_local, contract + '_clean')



# test 
# from trademl.modeling.structural_breaks import my_get_sadf
# from mlfinlab.structural_breaks import get_sadf
# import time

# close = data.close[:600]
# # mlfinlab
# start = time.time()
# get_sadf(close, 'linear', 1, 50, True, num_threads = 1)
# end = time.time()
# print(end - start)
# # my way
# start = time.time()
# my_get_sadf(close, 'linear', 1, 50, True)
# end = time.time()
# print(end - start)
