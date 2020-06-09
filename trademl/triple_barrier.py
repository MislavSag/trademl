# fundamental modules
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
import joblib
import json
import sys
# preprocessing
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier
from sklearn.base import clone
import xgboost
import shap
# metrics 
import mlfinlab as ml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    log_loss,
    )
from boruta import BorutaPy
# finance packagesb
import mlfinlab as ml
import trademl as tml
import vectorbt as vbt


### GLOBALS
DATA_PATH = 'D:/market_data/usa/ohlcv_features/'


### IMPORT DATA
contract = ['SPY']
with pd.HDFStore(DATA_PATH + contract[0] + '.h5') as store:
    data = store.get(contract[0])
data.sort_index(inplace=True)


### CHOOSE/REMOVE VARIABLES
remove_ohl = ['open', 'low', 'high', 'average', 'barCount',
              'vixFirst', 'vixHigh', 'vixLow', 'vixClose',
              'vixVolume', 'open_orig', 'high_orig', 'low_orig']
remove_ohl = [col for col in remove_ohl if col in data.columns]
data.drop(columns=remove_ohl, inplace=True)  #correlated with close
data['close_orig'] = data['close']  # with original close reslts are pretty bad!


### NON-MODEL HYPERPARAMETERS
std_outlier = 10
tb_volatility_lookback = 50
tb_volatility_scaler = 2
tb_triplebar_num_days = 90
tb_triplebar_pt_sl = [3, 3]
tb_triplebar_min_ret = 0.005
sample_weights_type = 'returns'
cv_type = 'purged_kfold'
cv_number = 4
max_features = 15
max_depth = 2
rand_state = 3
n_estimators = 1000
remove_ind_with_high_period = True
keep_important_features = 25
vectorbt_slippage = 0.0015
vectorbt_fees = 0.0015


### REMOVE INDICATORS WITH HIGH PERIOD
if remove_ind_with_high_period:
    data.drop(columns=['DEMA96000', 'ADX96000', 'TEMA96000',
                       'ADXR96000', 'TRIX96000'], inplace=True)
    data.drop(columns=['autocorr_1', 'autocorr_2', 'autocorr_3',
                       'autocorr_4', 'autocorr_5'], inplace=True)
    print('pass')
    

### REMOVE OUTLIERS
outlier_remove = tml.modeling.pipelines.OutlierStdRemove(std_outlier)
data = outlier_remove.fit_transform(data)

### TRIPLE BARRIERS TRUE
triple_barrier_pipe= tml.modeling.pipelines.TripleBarierLabeling(
    close_name='close_orig',
    volatility_lookback=tb_volatility_lookback,
    volatility_scaler=tb_volatility_scaler,
    triplebar_num_days=tb_triplebar_num_days,
    triplebar_pt_sl=tb_triplebar_pt_sl,
    triplebar_min_ret=tb_triplebar_min_ret,
    num_threads=1
)
tb_fit = triple_barrier_pipe.fit(data)
X = tb_fit.transform(data)

# triple barrier
daily_vol = ml.util.get_daily_vol(data.close_orig, lookback=tb_volatility_lookback)
cusum_events = ml.filters.cusum_filter(data.close_orig, threshold=daily_vol.mean()*tb_volatility_scaler)

# 
vertical_barriers = ml.labeling.add_vertical_barrier(
    t_events=cusum_events, close=data.close_orig, num_days=tb_triplebar_num_days) 


close = data.close_orig
t_events = cusum_events
pt_sl = tb_triplebar_pt_sl
min_ret = tb_triplebar_min_ret
target = daily_vol
num_threads = 1
vertical_barrier_times = vertical_barriers
side_prediction=None

triple_barrier_events = ml.labeling.get_events(
    close=close,
    t_events=cusum_events,
    pt_sl=tb_triplebar_pt_sl,
    target=daily_vol,
    min_ret=tb_triplebar_min_ret,
    num_threads=1,
    vertical_barrier_times=vertical_barriers)


# 1) Get target
target = target.reindex(t_events)
target = target[target > min_ret]  # min_ret

# 2) Get vertical barrier (max holding period)
if vertical_barrier_times is False:
    vertical_barrier_times = pd.Series(pd.NaT, index=t_events)
    
# 3) Form events object, apply stop loss on vertical barrier
if side_prediction is None:
    side_ = pd.Series(1.0, index=target.index)
    pt_sl_ = [pt_sl[0], pt_sl[0]]
else:
    side_ = side_prediction.reindex(target.index)  # Subset side_prediction on target index.
    pt_sl_ = pt_sl[:2]
    
# Create a new df with [v_barrier, target, side] and drop rows that are NA in target
events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_}, axis=1)
events = events.dropna(subset=['trgt'])

### apply_pt_sl_on_t1 
# Apply stop loss/profit taking, if it takes place before t1 (end of event)
events_ = events.copy()
out = events_[['t1']].copy(deep=True)

profit_taking_multiple = pt_sl[0]
stop_loss_multiple = pt_sl[1]

# Profit taking active
if profit_taking_multiple > 0:
    profit_taking = profit_taking_multiple * events_['trgt']
else:
    profit_taking = pd.Series(index=events.index)  # NaNs

# Stop loss active
if stop_loss_multiple > 0:
    stop_loss = -stop_loss_multiple * events_['trgt']
else:
    stop_loss = pd.Series(index=events.index)  # NaNs

vertical_barriers_filled = events_['t1'].fillna(close.index[-1]).values
for vb in vertical_barriers_filled:
    closing_prices = close[loc: vertical_barrier]  # Path prices for a given trade


for loc, vertical_barrier in events_['t1'].fillna(close.index[-1]).iteritems():
    print(loc)
    # closing_prices = close[loc: vertical_barrier]  # Path prices for a given trade
    # cum_returns = (closing_prices / close[loc] - 1) * events_.at[loc, 'side']  # Path returns
    # out.loc[loc, 'sl'] = cum_returns[cum_returns < stop_loss[loc]].index.min()  # Earliest stop loss date
    # out.loc[loc, 'pt'] = cum_returns[cum_returns > profit_taking[loc]].index.min()  # Earliest profit taking date




# Snippet 3.3 -> 3.6 page 50, Getting the Time of the First Touch, with Meta Labels
def get_events(close, t_events, pt_sl, target, min_ret, num_threads, vertical_barrier_times=False,
               side_prediction=None):
    # 1) Get target
    target = target.reindex(t_events)
    target = target[target > min_ret]  # min_ret

    # 2) Get vertical barrier (max holding period)
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events)

    # 3) Form events object, apply stop loss on vertical barrier
    if side_prediction is None:
        side_ = pd.Series(1.0, index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        side_ = side_prediction.reindex(target.index)  # Subset side_prediction on target index.
        pt_sl_ = pt_sl[:2]

    # Create a new df with [v_barrier, target, side] and drop rows that are NA in target
    events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_}, axis=1)
    events = events.dropna(subset=['trgt'])

    # Apply Triple Barrier
    first_touch_dates = mp_pandas_obj(func=apply_pt_sl_on_t1,
                                      pd_obj=('molecule', events.index),
                                      num_threads=num_threads,
                                      close=close,
                                      events=events,
                                      pt_sl=pt_sl_)

    for ind in events.index:
        events.loc[ind, 't1'] = first_touch_dates.loc[ind, :].dropna().min()

    if side_prediction is None:
        events = events.drop('side', axis=1)

    # Add profit taking and stop loss multiples for vertical barrier calculations
    events['pt'] = pt_sl[0]
    events['sl'] = pt_sl[1]

    return events



# Snippet 3.2, page 45, Triple Barrier Labeling Method
def apply_pt_sl_on_t1(close, events, pt_sl, molecule):  # pragma: no cover
    # Apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)

    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]

    # Profit taking active
    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_['trgt']
    else:
        profit_taking = pd.Series(index=events.index)  # NaNs

    # Stop loss active
    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_['trgt']
    else:
        stop_loss = pd.Series(index=events.index)  # NaNs

    # Get events
    for loc, vertical_barrier in events_['t1'].fillna(close.index[-1]).iteritems():
        closing_prices = close[loc: vertical_barrier]  # Path prices for a given trade
        cum_returns = (closing_prices / close[loc] - 1) * events_.at[loc, 'side']  # Path returns
        out.loc[loc, 'sl'] = cum_returns[cum_returns < stop_loss[loc]].index.min()  # Earliest stop loss date
        out.loc[loc, 'pt'] = cum_returns[cum_returns > profit_taking[loc]].index.min()  # Earliest profit taking date

    return out