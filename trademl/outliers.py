import pandas as pd


def remove_ohlc_ouliers(data, threshold_up=1, threshold_down=-0.3):
    """
    Remove ohlc obervations where value grater than threshold.
    That is, remove rows where growth grater than threshold.

    Returns:
        pd.DataFrame
    """
    ohlcCols = ["open", 'high', 'low', 'close']
    data = data.loc[
        (data[ohlcCols].pct_change(1) < threshold_up).all(1) &
        (data[ohlcCols].pct_change(1).shift(-1) > threshold_down).all(1)
    ]
    data = data.loc[
        (data[ohlcCols].pct_change(1) > threshold_down).all(1) &
        (data[ohlcCols].pct_change(1).shift(-1) < threshold_up).all(1)
    ]

    return data