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
