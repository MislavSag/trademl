import os
import pandas as pd
from pathlib import Path
import trademl as tml
from trademl.modeling.utils import time_method
import ib_insync
from ib_insync import *
util.startLoop()  # uncomment this line when in a notebook


@time_method
def import_ohlcv(path, contract='SPY_IB'):
    cache_path = os.path.join(Path(path), 'cache', contract + '.h5')
    if os.path.exists(cache_path):
        data = pd.read_hdf(cache_path, contract)
        q = 'SELECT date, open, high, low, close, volume, average, barCount FROM ' + contract + ' ORDER BY id DESC LIMIT 1'
        data_check = tml.modeling.utils.query_to_db(q, 'odvjet12_market_data_usa')
        if not (data_check['date'] == data.index[-1])[0]:        
            q = 'SELECT date, open, high, low, close, volume, average, barCount FROM ' + contract
            data = tml.modeling.utils.query_to_db(q, 'odvjet12_market_data_usa')
            data.set_index(data.date, inplace=True)
            data.drop(columns=['date'], inplace=True)
            data = data.sort_index()
            data.to_hdf(cache_path, contract)
    else:
        q = 'SELECT date, open, high, low, close, volume, average, barCount FROM ' + contract
        data = tml.modeling.utils.query_to_db(q, 'odvjet12_market_data_usa')
        data.set_index(data.date, inplace=True)
        data.drop(columns=['date'], inplace=True)
        data = data.sort_index()
        data.to_hdf(cache_path, contract)
     
    return data


import os
from pathlib import Path
import glob
import shutil
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import datetime
import time
import json
from utils import query_to_db
from utils import first_column_to_names
from utils import write_to_db


tickers = ['SPY', 'AAPL']
date_from = pd.Timestamp('2000-01-01 23:59:59')
date_to = datetime.datetime.now()
bar_size='1 hour'

   

    
    
test = ib_import(tickers=['SPY', 'AAPL'],
                 endDateTime=datetime.datetime.now(),
                 durationStr='1 Y',
                 barSizeSetting='1 day',
                 whatToShow='TRADES',
                 useRTH=True
                 )


    # get data for every contract
    while date_to > date_from:
        print(date_to)
        date_to = date_to.strftime('%Y%m%d %H:%M:%S')
        try:
            bars = ib.reqHistoricalData(
                contracts[i],
                endDateTime=date_to,
                durationStr='5 Y',
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1,
                timeout=60*10
            )
            df = util.df(bars)
            date_to = df['date'].min()
        except TypeError as te:
            print(te)
            break

    # Clean scraped tables
    files = glob.glob(save_path_prefix + '*')
    market_data = [pd.read_csv(f, sep=';', parse_dates=['date']) for f in files]
    market_data = pd.concat(market_data, axis=0)
    market_data['ticker'] = tickers[i]
    market_data.drop_duplicates(inplace=True)
    market_data = market_data.sort_values('date')
    
    # disconnect interactive brokers
    ib.disconnect()

    return market_data


# GLOBALS
dir_to_save = Path('D:/market_data/usa/ib_csv')

# connection
ib = IB()   
ib.connect('127.0.0.1', 7496, clientId=1)




# save final table to db and hdf5 file
write_to_db(market_data, "odvjet12_market_data_usa", con_list[i] + '_IB')
store_path = os.path.join(Path('D:/market_data/usa/'), con_list[i] + '_IB' + '.h5')
market_data.to_hdf(store_path, key=con_list[i])

# delete csv files
shutil.rmtree(dir_to_save)

