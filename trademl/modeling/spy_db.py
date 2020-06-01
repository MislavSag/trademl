# fundamental modules
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
# my functions
from outliers import remove_ohlc_ouliers
# tws
import ib_insync
from ib_insync import *
util.startLoop()  # uncomment this line when in a notebook



### GLOBAL (CONFIGS)

SPY_DATA_PATH = 'C:/Users/Mislav/algoAItrader/data/'


### PANDAS OPTIONS

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#### IMPORT RAW SPY DATA, REMOVE NA VALUES AND MERGE WITH IB SPY DATA

# import raw SPY data
spyRaw = pd.read_csv(
    SPY_DATA_PATH + 'SpyVixWithIndicators.csv', sep=';', decimal=',',
    usecols=list(range(59)),parse_dates=['TimeBarStart'], 
    index_col='TimeBarStart'
    )

# vix
vox_columns = ['VixFirstTradePrice', 'VixHighTradePrice', 
                'VixLowTradePrice', 'VixLastTradePrice', 'VixVolume']
vix = spyRaw[vox_columns]
vix.columns = ['vixFirst', 'vixHigh', 'vixLow', 'vixClose', 'vixVolume']

# keep OHLC data, remove other
spy = spyRaw[['SpyFirstTradePrice', 'SpyHighTradePrice', 'SpyLowTradePrice',
            'SpyLastTradePrice', 'SpyVolume']]
spy.columns = ['open', 'high', 'low', 'close', 'volume']

# missing data spy
print(spy.isna().sum())
spy = spy.dropna(subset=['close'])
print(spy.isna().sum())

# missing data vix
print("Missing vlaues by cols \n", vix.isna().sum())
print("Share of missing values in spy \n", vix.isna().sum().iloc[0] / spyRaw.shape[0])  # NA share in spy
isnaTime = vix.isna().any(1).index.date
isnaTime = (pd.DataFrame(isnaTime, columns=['time']).
            groupby(['time']).size().sort_values(ascending=False))
print(isnaTime.head(10))
vix.loc['1998-01-21 09:00:00':'1998-01-21 12:00:00']

# merge spy and vix with merge_asof which uses nearest back value for NA
spy = spy.sort_index()
vix = vix.sort_index()
vix = vix.dropna()
spy = pd.merge_asof(spy, vix, left_index=True, right_index=True)
print(spy.shape)
print(spy.shape)
print(spy.head())
print(spy.head())
spy.loc['1998-01-21 09:00:00':'1998-01-21 13:10:00']
spy.dropna(inplace=True)  # remove NA values in VIX from the begining
spy = spy.sort_index()

# get collected IB data and merge with spy data


def clean_ib_tables(spy, blob_path='/spy_*csv'):
    df = glob.glob(SPY_DATA_PATH + '/spy_*csv')
    df = [pd.read_csv(path, sep=';', index_col=['date'], parse_dates=['date'])
           for path in df]
    df = pd.concat(df, axis=0)
    df.drop_duplicates(keep='first', inplace=True)
    new_dates = np.setdiff1d(df.index.date, spy.index.date)
    df['datetime'] = df.index
    df.set_index(df.index.date, inplace=True)
    df = df.loc[df.index.isin(new_dates)]
    df.set_index(df['datetime'], inplace=True)
    df.drop(columns=['datetime'], inplace=True)
    
    return df


# merge new vix, new spy and old spy
spy_new = clean_ib_tables(spy)
vix_new = clean_ib_tables(spy, '/vix_*csv')
vix_new.rename(columns={'open': 'vixFirst', 'high': 'vixHigh', 'low': 
    'vixLow', 'close': 'vixClose', 'volume': 'vixVolume', 'average': 'vixAverage',
    'barCount': 'vixBarCount'}, inplace=True)
spy_vix = pd.concat([vix_new, spy_new], axis=1)
spy = pd.concat([spy, spy_vix], axis=0)



### GET RESIDUAL SPY DATA FROM IB

# connection
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=3)

# contract
# contract = Stock('SPY', 'SMART', 'USD')
contract = Index("vix")
contract = ib.qualifyContracts(contract)[0]


### KEEP THIS FUNCTION FOR LATER WHEN I WILL NEED IT FOR STOCK MARKET DATA ###

# while loop which retrieve historical data
# date_end_loop = pd.Timestamp('2003-12-31 23:59:59')
# max_date = datetime.datetime(2020, 4, 23, 23, 59, 59, 99)
# while max_date > date_end_loop:
#     max_date = max_date.strftime('%Y%m%d %H:%M:%S')
#     bars = ib.reqHistoricalData(
#         contract,
#         endDateTime=max_date,
#         durationStr='6 M',
#         barSizeSetting='1 min',
#         whatToShow='TRADES',
#         useRTH=True,
#         formatDate=1
#     )
#     df = util.df(bars)
#     max_date = df['date'].min()
#     df.to_csv(SPY_DATA_PATH + '/vix_' + max_date.strftime('%Y-%m-%d') + '.csv', index=False, sep=';')

### KEEP THIS FUNCTION FOR LATER WHEN I WILL NEED IT FOR STOCK MARKET DATA ###

# get newest data
# spy.index.max()  # last date
# bars = ib.reqHistoricalData(
#     contract,
#     endDateTime='',
#     durationStr='2 M',
#     barSizeSetting='1 min',
#     whatToShow='TRADES',
#     useRTH=True,
#     formatDate=1
# )
# df = util.df(bars)
# df.to_csv(SPY_DATA_PATH + '/spy_' + '2020-04-23' + '.csv', index=False, sep=';')

# disconnect from IB
# ib.disconnect()


# there are outliers
spyPriceReturnsPLot = pd.concat([spy['close'],
                                 spy['close'].pct_change(1, freq='Min')],
                                axis=1)
spyPriceReturnsPLot.columns = ['last', 'returns']
spyPriceReturnsPLot.plot(subplots=True)

# remove outliers
spy = remove_ohlc_ouliers(spy, threshold_up=0.20, threshold_down=-0.20)


### DESCRIPTIVE ANALYSIS

# # last price plot and returns plot
# spyPriceReturnsPLot = pd.concat([spy['close'],
#                                 spy['close'].pct_change(1)],
#                                 axis=1)
# spyPriceReturnsPLot.columns = ['last', 'returns']
# spyPriceReturnsPLot.plot(subplots=True)

# # returns histogram
# spy['close'].pct_change(1).plot(kind='hist', bins=1000, xlim=(-0.01, 0.01))
# sns.distplot(spy['close'].pct_change(1).dropna())

# # cumulative return plot
# spy['close'].pct_change(1).cumsum().plot(title='Cummulative returns')

# summarizes
with pd.option_context('float_format', '{:f}'.format):
    print(spy['close'].describe())
with pd.option_context('float_format', '{:f}'.format):
    print(spy['close'].pct_change(1, freq='Min').describe())


# SAVE AS HDF5 FILE
# HDF5: Hierarchical data format, developed initially at the National Center
# for Supercomputing, is a fast and scalable storage format for numerical data,
# available in pandas using the PyTables library.
spy_store = Path(SPY_DATA_PATH + 'spy.h5')
with pd.HDFStore(spy_store) as store:
    store.put('spy', spy)
