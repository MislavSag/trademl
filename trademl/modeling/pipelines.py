import numpy as np 
import pandas as pd
from numba import njit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import mlfinlab as ml
from trademl.modeling.utils import time_method


class TripleBarierLabeling(BaseEstimator, TransformerMixin):

    def __init__(self, close_name='close', volatility_lookback=50,
                 volatility_scaler=1, triplebar_num_days=5,
                 triplebar_pt_sl=[1, 1], triplebar_min_ret=0.003,
                 num_threads=1, tb_min_pct=0.05):
        # hyperparameters for all functions
        self.close_name = close_name
        self.volatility_lookback = volatility_lookback
        self.volatility_scaler = volatility_scaler
        self.triplebar_num_days = triplebar_num_days
        self.triplebar_pt_sl = triplebar_pt_sl
        self.triplebar_min_ret = triplebar_min_ret
        self.num_threads = num_threads
        self.min_pct = tb_min_pct

    @time_method
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
        labels = ml.labeling.drop_labels(labels, self.min_pct)
        
        # merge labels and triple barrier events
        self.triple_barrier_info = pd.concat([triple_barrier_events.t1, labels], axis=1)
        self.triple_barrier_info.dropna(inplace=True)
        
        return self
    
    @time_method
    def transform(self, X, y=None):
        
        # subsample
        X = X.reindex(self.triple_barrier_info.index)
        
        return X



class OutlierStdRemove(BaseEstimator, TransformerMixin):

    def __init__(self, std_threshold):
        self.std_threshold = std_threshold

    @time_method
    def fit(self, X, y=None):
        return self

    @time_method
    def transform(self, X, y=None):
        X = X[X.apply(lambda x: (np.abs(x - x.mean()) / x.std()) < self.std_threshold).
              all(axis=1)]
        return X


@njit
def calculate_t_values(subset, min_sample_length, step):  # pragma: no cover
    """
    For loop for calculating linear regression every n steps.

    :param subset: (np.array) subset of indecies for which we want to calculate t values
    :return: (float) maximum t value and index of maximum t value
    """
    max_abs_t_value = -np.inf  # Maximum abs t-value of b_1 coefficient among l values
    max_t_value_index = None  # Index with maximum t-value

    for forward_window in np.arange(min_sample_length, subset.shape[0], step):

        y_subset = subset[:forward_window].reshape(-1, 1)  # y{t}:y_{t+l}

        # Array of [1, 0], [1, 1], [1, 2], ... [1, l] # b_0, b_1 coefficients
        x_subset = np.ones((y_subset.shape[0], 2))
        x_subset[:, 1] = np.arange(y_subset.shape[0])

        # Get regression coefficients estimates
        xy_ = x_subset.transpose() @ y_subset
        xx_ = x_subset.transpose() @ x_subset

        #   check for singularity
        det = np.linalg.det(xx_)

        # get coefficient and std from linear regression
        if det == 0:
            b_mean = np.array([[np.nan]])
            b_std = np.array([[np.nan, np.nan]])
        else:
            xx_inv = np.linalg.inv(xx_)
            b_mean = xx_inv @ xy_
            err = y_subset - (x_subset @ b_mean)
            b_std = np.dot(np.transpose(err), err) / (x_subset.shape[0] - x_subset.shape[1]) * xx_inv  # pylint: disable=E1136  # pylint/issues/3139

        # Check if l gives the maximum t-value among all values {0...L}
            t_beta_1 = (b_mean[1] / np.sqrt(b_std[1, 1]))[0]
            if abs(t_beta_1) > max_abs_t_value:
                max_abs_t_value = abs(t_beta_1)
                max_t_value = t_beta_1
                max_t_value_index = forward_window

    return max_t_value_index, max_t_value


def trend_scanning_labels(price_series: pd.Series, t_events: list = None, look_forward_window: int = 20,
                          min_sample_length: int = 5, step: int = 1) -> pd.DataFrame:
    """
    `Trend scanning <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419>`_ is both a classification and
    regression labeling technique.

    That can be used in the following ways:

    1. Classification: By taking the sign of t-value for a given observation we can set {-1, 1} labels to define the
       trends as either downward or upward.
    2. Classification: By adding a minimum t-value threshold you can generate {-1, 0, 1} labels for downward, no-trend,
       upward.
    3. The t-values can be used as sample weights in classification problems.
    4. Regression: The t-values can be used in a regression setting to determine the magnitude of the trend.

    The output of this algorithm is a DataFrame with t1 (time stamp for the farthest observation), t-value, returns for
    the trend, and bin.

    :param price_series: (pd.Series) close prices used to label the data set
    :param t_events: (list) of filtered events, array of pd.Timestamps
    :param look_forward_window: (int) maximum look forward window used to get the trend value
    :param min_sample_length: (int) minimum sample length used to fit regression
    :param step: (int) optimal t-value index is searched every 'step' indices
    :return: (pd.DataFrame) of t1, t-value, ret, bin (label information). t1 - label endtime, tvalue,
        ret - price change %, bin - label value based on price change sign
    """
    # pylint: disable=invalid-name

    if t_events is None:
        t_events = price_series.index

    t1_array = []  # Array of label end times
    t_values_array = []  # Array of trend t-values

    for index in t_events:
        subset = price_series.loc[index:].iloc[:look_forward_window]  # Take t:t+L window
        if subset.shape[0] >= look_forward_window:

            # linear regressoin for every index
            max_t_value_index, max_t_value = calculate_t_values(subset.values,
                                                                min_sample_length,
                                                                step)

            # Store label information (t1, return)
            label_endtime_index = subset.index[max_t_value_index - 1]
            t1_array.append(label_endtime_index)
            t_values_array.append(max_t_value)

        else:
            t1_array.append(None)
            t_values_array.append(None)

    labels = pd.DataFrame({'t1': t1_array, 't_value': t_values_array}, index=t_events)
    labels.loc[:, 'ret'] = price_series.reindex(labels.t1).values / price_series.reindex(labels.index).values - 1
    labels['bin'] = np.sign(labels.t_value)

    return labels



class TrendScanning(BaseEstimator, TransformerMixin):

    def __init__(self, close_name='close', volatility_lookback=50,
                 volatility_scaler=1, ts_look_forward_window=20, # 4800,  # 60 * 8 * 10 (10 days)
                 ts_min_sample_length=5, ts_step=1):
        self.close_name = close_name
        self.volatility_lookback = volatility_lookback
        self.volatility_scaler = volatility_scaler
        self.ts_look_forward_window = ts_look_forward_window
        self.ts_min_sample_length = ts_min_sample_length
        self.ts_step = ts_step
        self.ts = None

    @time_method
    def fit(self, X, y=None):

        # extract close series
        close = X.loc[:, self.close_name]

        # Compute volatility
        daily_vol = ml.util.get_daily_vol(
            close,
            lookback=self.volatility_lookback)

        # Apply Symmetric CUSUM Filter and get timestamps for events
        cusum_events = ml.filters.cusum_filter(close,
            threshold=daily_vol.mean()*self.volatility_scaler)

        # get trend scanning labels
        trend_scanning = trend_scanning_labels(
            close, 
            t_events=cusum_events,
            look_forward_window=20,
            min_sample_length=5,
            step=1)
        trend_scanning.dropna(inplace=True)

        self.ts = trend_scanning

        return self.ts

    @time_method
    def transform(self, X, y=None):

        # subsample
        X = X.reindex(self.ts.index)

        return X





#################### TESTS


# DATA_PATH = 'C:/Users/Mislav/algoAItrader/data/spy_with_vix.h5'
# df = pd.read_hdf(DATA_PATH, start=0, stop=10000)


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


## TREND SCANNING
# close_name='close_orig'
# tb_volatility_lookback = 50
# tb_volatility_scaler = 1
# ts_look_forward_window=20
# ts_min_sample_length=5
# ts_step=1
    # trend_scanning_pipe = TrendScanning(
    #     close_name, ts_look_forward_window, ts_min_sample_length, ts_step
    #     )
    # ts_fit = trend_scanning_pipe.fit(df)
    # X = trend_scanning_pipe.transform(df)
#################### TESTS
