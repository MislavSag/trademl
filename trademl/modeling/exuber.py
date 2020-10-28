import pandas as pd
import numpy as np
from pathlib import Path
import os
from mlfinlab.structural_breaks import get_sadf
from trademl.modeling.data_import import import_ohlcv
from sklearn.pipeline import make_pipeline
from trademl.modeling.outliers import RemoveOutlierDiffMedian


# Parameters
save_path = 'D:/market_data/usa/ohlcv_features'
contract = 'SPY_IB'
keep_unstationary = True

# Import data
data = import_ohlcv(save_path, contract=contract)

# Remove outliers
pipe = make_pipeline(
    RemoveOutlierDiffMedian(median_outlier_thrteshold=25)
    )
X = pipe.fit_transform(data)

# prepare close
close = X['close']
close_daily = close.resample('D').last().dropna()
close_hourly = close.resample('H').last().dropna()

#  rolling apply sadf
def get_last_sadf(close, model='linear'):
    radf = get_sadf(
        series=close,
        model=model,
        lags=3,
        min_length=50,
        add_const=True,
        # phi
        num_threads=1,
        verbose=True
    )
    return radf[-1]

# rolling 
win_length = 150
print('Start sadf with linear model')
radf_d_l = close_daily.rolling(win_length).apply(lambda x: get_last_sadf(x, model='linear'))
print('Start sadf with quadratic model')
radf_d_q = close_daily.rolling(win_length).apply(lambda x: get_last_sadf(x, model='quadratic'))
print('Start sadf with sm_poly_1 model')
radf_d_p1 = close_daily.rolling(win_length).apply(lambda x: get_last_sadf(x, model='sm_poly_1'))
print('Start sadf with sm_exp model')
radf_d_e = close_daily.rolling(win_length).apply(lambda x: get_last_sadf(x, model='sm_exp'))
print('Start sadf with sm_power model')
radf_d_p = close_daily.rolling(win_length).apply(lambda x: get_last_sadf(x, model='sm_power'))
sadf_d = pd.concat([radf_d_l, radf_d_q, radf_d_p1, radf_d_e, radf_d_p], axis=1)
sadf_d.columns = ['linear', 'quadratic', 'sm_poly_1', 'sm_exp', 'sm_power']

# save
save_path = Path('D:/algo_trading_files/exuber')
sadf_d.to_csv(os.path.join(save_path, 'sadf_py.csv'))

# inspect results
# save_path = Path('D:/algo_trading_files/exuber')
# df = pd.read_csv(os.path.join(save_path, 'sadf_py.csv'))
# df.index = df['date']
# df = df.drop(columns=['date'])
# df = df.dropna()
# df[['linear']].plot()
# df[['quadratic']].plot()
# df[['sm_poly_1']].plot()
# df[['sm_exp']].plot()
# df[['sm_power']].plot()
# radf_1 = close_daily[:601].rolling(600).apply(lambda x: get_last_sadf(x, model='linear'))
# radf_2 = close_daily[500:601].rolling(100).apply(lambda x: get_last_sadf(x, model='linear'))
# radf_1.dropna()
# radf_2.dropna()
