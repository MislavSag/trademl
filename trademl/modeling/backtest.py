import pandas as pd
import numpy as np
import numba
from numba import njit


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


@njit
def enter_positions(positions):
    for i in range(positions.shape[0]): # postitions.shape[0]
        if i > 0:
            if (positions[i, 1] == -1):
                positions[i, 1] = -1
            elif (positions[i-1, 1] == -1) & (np.isnan(positions[i, 1])):
                positions[i, 1] = -1
            else: 
                positions[i, 1] = 1
        else:
            positions[i, 1] = 1
    return positions


def hold_cash_backtest(close, signs):
    """
    Backtest strategy that has only two positions; holding asset or cash position.

    :param close: (pd.seires) of true close prices
    :param signs: (pd.Series) of signs (1 - holding, 0 -cash) 
    :param result: ()
    """
    # true cumulative returns
    close_cum = cumulative_returns(close)

    # concat close true and predictions
    positions = pd.concat([close, signs.rename('position')], axis=1)
    
    # enter posistions for all dates: nter -1 untill 1 appears, vice versa
    predictions = enter_positions(positions.values)
    predictions = pd.DataFrame(predictions, index=close.index, columns=['close', 'position'])
    
    # calculate returns of the strategy
    predictions['adjusted_close'] = np.where(predictions.position == 1, predictions.close, np.nan)
    predictions['return'] = predictions['adjusted_close'].pct_change(fill_method=None)
    predictions['cum_return'] = (1 + predictions['return']).cumprod()
    true_vs_pred = pd.concat([close_cum, predictions], axis=1)

    return true_vs_pred
