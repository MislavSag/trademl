import pandas as pd
import numpy as np


def cumulative_returns(close, raw=True):
    """
    Calculate cumulative returns
    param close: (pd.series) of close prices or returns
    param raw:
    """
    # perfrormance
    if raw:
        close = close.pct_change()
    close = (1 + close).cumprod()
    return close.dropna()


def cumulative_returns_tb(trbar_info, predictions, bet_sizes=None, time_index=True):
    """
    
    """
    
    return_adj = np.where(
        predictions == trbar_info['bin'],
        trbar_info['ret'].abs(), -(trbar_info['ret'].abs()))
    if bet_sizes is not None:
        return_adj = return_adj * bet_sizes
    if time_index:
        return_adj = pd.Series(return_adj, index=trbar_info.t1) #.to_frame()
        return_adj.index.name = None
    else:
        return_adj = pd.Series(return_adj)
    perf = cumulative_returns(return_adj, raw=False)
    return perf.rename('cumulative_return', inplace=True)


def minute_to_daily_return(trbar_info, predictions, bet_sizes=None):
    return_adj = np.where(
        predictions == trbar_info['bin'],
        trbar_info['ret'].abs(), -(trbar_info['ret'].abs()))
    if bet_sizes is not None:
        return_adj = return_adj * bet_sizes
    return_adj = pd.Series(return_adj, index=trbar_info.t1)
    daily_returns = (1 + return_adj).resample('B').prod() - 1
    return daily_returns