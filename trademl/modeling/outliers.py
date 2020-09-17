import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def remove_ourlier_diff_median(data, median_scaler=25):
    """
    Remove outliers by removing observations where differene is grater than
    daiy difference times the median scaler.

    :param data: (pd.DataFrame) with ohlc data
    :return: (pd.DataFrame) with removed outliers
    """
    daily_diff = (data.resample('D').last().dropna().diff().abs() + 0.05) * median_scaler
    daily_diff['diff_date'] = daily_diff.index.strftime('%Y-%m-%d')
    data_test = data.diff()
    data_test['diff_date'] = data_test.index.strftime('%Y-%m-%d')
    data_test_diff = pd.merge(data_test, daily_diff, on='diff_date')
    indexer = ((np.abs(data_test_diff['close_x']) < np.abs(data_test_diff['close_y'])) &
            (np.abs(data_test_diff['open_x']) < np.abs(data_test_diff['open_y'])) &
            (np.abs(data_test_diff['high_x']) < np.abs(data_test_diff['high_y'])) & 
            (np.abs(data_test_diff['low_x']) < np.abs(data_test_diff['low_y'])))
    # indexer = (indexer | data_test_diff['close_y'].isna())
    data_final = data.loc[indexer.values, :]
    
    return data_final


class RemoveOutlierDiffMedian(BaseEstimator, TransformerMixin):

    def __init__(self, median_outlier_thrteshold, state={}):
        self.median_outlier_thrteshold = median_outlier_thrteshold
        self.state = state

    def fit(self, X, y=None, state={}):
        if type(X) is tuple: X, y, self.state = X
        
        print(f"Removing outliers")
        
        return self

    def transform(self, X, y=None, state={}):
        if type(X) is tuple: X, y, self.state = X
        
        print(f"Shape before outlier removal: {X.shape}")
        
        X = remove_ourlier_diff_median(X, self.median_outlier_thrteshold)
        
        print(f"Shape after outlier removal: {X.shape}")

        return X
