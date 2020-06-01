import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import mlfinlab as ml


class TripleBarierLabeling(BaseEstimator, TransformerMixin):

    def __init__(self, close_name='close', volatility_lookback=50,
                 volatility_scaler=1, triplebar_num_days=5,
                 triplebar_pt_sl=[1, 1], triplebar_min_ret=0.003,
                 num_threads=1):
        # hyperparameters for all functions
        self.close_name = close_name
        self.volatility_lookback = volatility_lookback
        self.volatility_scaler = volatility_scaler
        self.triplebar_num_days = triplebar_num_days
        self.triplebar_pt_sl = triplebar_pt_sl
        self.triplebar_min_ret = triplebar_min_ret
        self.num_threads = num_threads

    def fit(self, X, y=None):
        
        # extract close series
        close = X.loc[:, self.close_name]
        
        # Compute volatility
        daily_vol = ml.util.get_daily_vol(
            close,
            lookback=self.volatility_lookback)
        
        # Apply Symmetric CUSUM Filter and get timestamps for events
        cusum_events = ml.filters.cusum_filter(
            close,
            threshold=daily_vol.mean()*self.volatility_scaler)
        
        # Compute vertical barrier
        vertical_barriers = ml.labeling.add_vertical_barrier(
            t_events=cusum_events,
            close=close,
            num_days=self.triplebar_num_days) 
        
        # tripple barier events
        triple_barrier_events = ml.labeling.get_events(
            close=close,
            t_events=cusum_events,
            pt_sl=self.triplebar_pt_sl,
            target=daily_vol,
            min_ret=self.triplebar_min_ret,
            num_threads=self.num_threads,
            vertical_barrier_times=vertical_barriers)
        
        # labels
        labels = ml.labeling.get_bins(triple_barrier_events, close)
        labels = ml.labeling.drop_labels(labels)
        
        # merge labels and triple barrier events
        self.triple_barrier_info = pd.concat([triple_barrier_events.t1, labels], axis=1)
        self.triple_barrier_info.dropna(inplace=True)
        
        return self

    def transform(self, X, y=None):
        
        # subsample
        X = X.reindex(self.triple_barrier_info.index)
        
        return X



class OutlierStdRemove(BaseEstimator, TransformerMixin):

    def __init__(self, std_threshold):
        self.std_threshold = std_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[X.apply(lambda x: np.abs(x - x.mean()) / x.std() < self.std_threshold).
              all(axis=1)]
        return X


### TESTS


# DATA_PATH = 'C:/Users/Mislav/algoAItrader/data/spy_with_vix.h5'
# df = pd.read_hdf(DATA_PATH, start=0, stop=4000)


# ### HYPER PARAMETERS
# std_outlier = 10
# tb_volatility_lookback = 50
# tb_volatility_scaler = 1
# tb_triplebar_num_days = 3
# tb_triplebar_pt_sl = [1, 1]
# tb_triplebar_min_ret = 0.003


# # triple barrier alone
# triple_barrier_pipe= TripleBarierLabeling(
#     close_name='close_orig',
#     volatility_lookback=tb_volatility_lookback,
#     volatility_scaler=tb_volatility_scaler,
#     triplebar_num_days=tb_triplebar_num_days,
#     triplebar_pt_sl=tb_triplebar_pt_sl,
#     triplebar_min_ret=tb_triplebar_min_ret,
#     num_threads=1
# )
# tb_fit = triple_barrier_pipe.fit(df)
# tb_fit.triple_barrier_info
# X = triple_barrier_pipe.transform(df)

# # 
# pipeline = Pipeline([
#     ('remove_outlier', OutlierStdRemove(10)),
#     ('triple_barrier_labeling', TripleBarierLabeling(close_name='close_orig')),
# ])

# pipe_out = pipeline.fit_transform(df)
