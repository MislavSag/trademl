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
            look_forward_window=self.ts_look_forward_window,
            min_sample_length=self.ts_min_sample_length,
            step=self.ts_step)
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
# df = pd.read_hdf(DATA_PATH, start=0, stop=1000000)


# # ### HYPER PARAMETERS
# std_outlier = 10
# tb_volatility_lookback = 50
# tb_volatility_scaler = 1
# tb_triplebar_num_days = 3
# tb_triplebar_pt_sl = [1, 1]
# tb_triplebar_min_ret = 0.003


# # INSPECT
# close = df['close_orig']
# daily_vol = ml.util.get_daily_vol(close,lookback=50)
# cusum_events = ml.filters.cusum_filter(close,threshold=daily_vol.mean())
# vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events,close=close,num_days=5) 

# # main function args
# t_events = cusum_events
# pt_sl=[1,1]
# target = daily_vol
# min_ret = 0.003
# vertical_barrier_times = vertical_barriers
# side_prediction = None



# # 1) Get target
# target = target.reindex(t_events)
# target = target[target > min_ret]  # min_ret

# # 2) Get vertical barrier (max holding period)
# if vertical_barrier_times is False:
#     vertical_barrier_times = pd.Series(pd.NaT, index=t_events, dtype=t_events.dtype)

# # 3) Form events object, apply stop loss on vertical barrier
# if side_prediction is None:
#     side_ = pd.Series(1.0, index=target.index)
#     pt_sl_ = [pt_sl[0], pt_sl[0]]
# else:
#     side_ = side_prediction.reindex(target.index)  # Subset side_prediction on target index.
#     pt_sl_ = pt_sl[:2]

# # Create a new df with [v_barrier, target, side] and drop rows that are NA in target
# events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_}, axis=1)
# events = events.dropna(subset=['trgt'])

# # _dates = mp_pandas_obj(func=apply_pt_sl_on_t1,
# #                                       pd_obj=('molecule', events.index),
# #                                       num_threads=num_threads,
# #                                       close=close,
# #                                       events=events,
# #                                       pt_sl=pt_sl_,
# #                                       verbose=verbose)


# # apply_pt_sl_on_t1_fast function
# molecule = events.index
# events=events
# pt_sl=[1,1]


# events_ = events.loc[molecule]
# out = events_[['t1']].copy(deep=True)

# profit_taking_multiple = pt_sl[0]
# stop_loss_multiple = pt_sl[1]

# # Profit taking active
# if profit_taking_multiple > 0:
#     profit_taking = profit_taking_multiple * events_['trgt']
# else:
#     profit_taking = pd.Series(index=events.index)  # NaNs

# # Stop loss active
# if stop_loss_multiple > 0:
#     stop_loss = -stop_loss_multiple * events_['trgt']
# else:
#     stop_loss = pd.Series(index=events.index)  # NaNs

# out['pt'] = pd.Series(dtype=events.index.dtype)
# out['sl'] = pd.Series(dtype=events.index.dtype)


# def df_to_numpy_with_index(df, column_name='datetime'):
#     df = df.to_frame()
#     df['column_name'] = df.index
#     df = df.values
#     return df


# close_val = close.values
# close_date = close.index.values
# loop_vec = df_to_numpy_with_index(events_['t1'].fillna(close.index[-1]))
# side_val = events_['side'].values
# stop_loss_val = stop_loss.values
# stop_loss_date = stop_loss.index.values
# profit_taking_val = profit_taking.values
# profit_taking_date = profit_taking.index.values

# # events_help = events_['t1'].fillna(close.index[-1])
# # close_indecies = np.where(events_help.index)


# @njit
# def apply_pt_sl_on_t1_fast(close_val, close_date, loop_vec, side_val, stop_loss_val, stop_loss_date, profit_taking_val, profit_taking_date):
#     sl_list = []
#     pt_list = []
#     for a in range(loop_vec.shape[0]):
#         t_1 = loop_vec[a, 1]
#         t_2 = loop_vec[a, 0]
#         closing_prices = close_val[np.where((close_date > t_1) & (close_date < t_2))]  # Path prices for a given trade
#         cum_returns = (closing_prices / close_val[np.where(close_date == t_1)[0]] - 1) * side_val[np.where(close_date == t_1)[0]]  # Path returns
#         sl_ = close_date[np.where(cum_returns[cum_returns < stop_loss_val[np.where(stop_loss_date == t_1)[0]]][0] == cum_returns)]
#         sl_list.append(sl_)
#         pt_ = close_date[np.where(cum_returns[cum_returns > profit_taking_val[np.where(profit_taking_date == t_1)[0]]][0] == cum_returns)]
#         pt_list.append(pt_)
#         # sl_list.append(t_1)
#         # pt_list.append(t_2)
#     return sl_list, pt_list

# sl_dates, pt_dates = apply_pt_sl_on_t1_fast(close_val, close_date, loop_vec, side_val, stop_loss_val, stop_loss_date, profit_taking_val, profit_taking_date)




# # out.loc[:, ['sl']] = np.array(sl_dates)
# # out.loc[:, ['pt']] = np.array(pt_dates)

# # # Get events
# # for loc, vertical_barrier in events_['t1'].fillna(close.index[-1]).iteritems():
# #     closing_prices = close[loc: vertical_barrier]  # Path prices for a given trade
# #     cum_returns = (closing_prices / close[loc] - 1) * events_.at[loc, 'side']  # Path returns
# #     out.at[loc, 'sl'] = cum_returns[cum_returns < stop_loss[loc]].index.min()  # Earliest stop loss date
# #     out.at[loc, 'pt'] = cum_returns[cum_returns > profit_taking[loc]].index.min()  # Earliest profit taking date



# # def apply_pt_sl_on_t1(close, events, pt_sl, molecule):  # pragma: no cover
# #     # Apply stop loss/profit taking, if it takes place before t1 (end of event)
# #     events_ = events.loc[molecule]
# #     out = events_[['t1']].copy(deep=True)

# #     profit_taking_multiple = pt_sl[0]
# #     stop_loss_multiple = pt_sl[1]

# #     # Profit taking active
# #     if profit_taking_multiple > 0:
# #         profit_taking = profit_taking_multiple * events_['trgt']
# #     else:
# #         profit_taking = pd.Series(index=events.index)  # NaNs

# #     # Stop loss active
# #     if stop_loss_multiple > 0:
# #         stop_loss = -stop_loss_multiple * events_['trgt']
# #     else:
# #         stop_loss = pd.Series(index=events.index)  # NaNs

# #     out['pt'] = pd.Series(dtype=events.index.dtype)
# #     out['sl'] = pd.Series(dtype=events.index.dtype)

# #     # Get events
# #     for loc, vertical_barrier in events_['t1'].fillna(close.index[-1]).iteritems():
# #         closing_prices = close[loc: vertical_barrier]  # Path prices for a given trade
# #         cum_returns = (closing_prices / close[loc] - 1) * events_.at[loc, 'side']  # Path returns
# #         out.at[loc, 'sl'] = cum_returns[cum_returns < stop_loss[loc]].index.min()  # Earliest stop loss date
# #         out.at[loc, 'pt'] = cum_returns[cum_returns > profit_taking[loc]].index.min()  # Earliest profit taking date

# #     return out



# # import numpy as np
# # from numba import njit


# # @njit
# # def datetime_operand(date1, date2):
# #     x = date1 > date2
# #     return x


# # d1 = np.array(['2001-01-01T12:00', '2002-02-03T13:56:03.172'],
# #               dtype='datetime64')
# # d2 = np.datetime64('2005-02-25T03:40')
# # print(datetime_operand(d1, d2))




# # # Snippet 3.3 -> 3.6 page 50, Getting the Time of the First Touch, with Meta Labels
# # def get_events(close, t_events, pt_sl, target, min_ret, num_threads, vertical_barrier_times=False,
# #                side_prediction=None, verbose=True):

# #     # 1) Get target
# #     target = target.reindex(t_events)
# #     target = target[target > min_ret]  # min_ret

# #     # 2) Get vertical barrier (max holding period)
# #     if vertical_barrier_times is False:
# #         vertical_barrier_times = pd.Series(pd.NaT, index=t_events, dtype=t_events.dtype)

# #     # 3) Form events object, apply stop loss on vertical barrier
# #     if side_prediction is None:
# #         side_ = pd.Series(1.0, index=target.index)
# #         pt_sl_ = [pt_sl[0], pt_sl[0]]
# #     else:
# #         side_ = side_prediction.reindex(target.index)  # Subset side_prediction on target index.
# #         pt_sl_ = pt_sl[:2]

# #     # Create a new df with [v_barrier, target, side] and drop rows that are NA in target
# #     events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_}, axis=1)
# #     events = events.dropna(subset=['trgt'])

# #     # Apply Triple Barrier
# #     first_touch_dates = mp_pandas_obj(func=apply_pt_sl_on_t1,
# #                                       pd_obj=('molecule', events.index),
# #                                       num_threads=num_threads,
# #                                       close=close,
# #                                       events=events,
# #                                       pt_sl=pt_sl_,
# #                                       verbose=verbose)

# #     for ind in events.index:
# #         events.at[ind, 't1'] = first_touch_dates.loc[ind, :].dropna().min()

# #     if side_prediction is None:
# #         events = events.drop('side', axis=1)

# #     # Add profit taking and stop loss multiples for vertical barrier calculations
# #     events['pt'] = pt_sl[0]
# #     events['sl'] = pt_sl[1]

# #     return events

# # # triple barrier alone
# # triple_barrier_pipe= TripleBarierLabeling(
# #     close_name='close_orig',
# #     volatility_lookback=tb_volatility_lookback,
# #     volatility_scaler=tb_volatility_scaler,
# #     triplebar_num_days=tb_triplebar_num_days,
# #     triplebar_pt_sl=tb_triplebar_pt_sl,
# #     triplebar_min_ret=tb_triplebar_min_ret,
# #     num_threads=1
# # )
# # tb_fit = triple_barrier_pipe.fit(df)
# # tb_fit.triple_barrier_info
# # X = triple_barrier_pipe.transform(df)

# # # 
# # pipeline = Pipeline([
# #     ('remove_outlier', OutlierStdRemove(10)),
# #     ('triple_barrier_labeling', TripleBarierLabeling(close_name='close_orig')),
# # ])

# # pipe_out = pipeline.fit_transform(df)


# ## TREND SCANNING
# # close_name='close_orig'
# # tb_volatility_lookback = 50
# # tb_volatility_scaler = 1
# # ts_look_forward_window=20
# # ts_min_sample_length=5
# # ts_step=1
#     # trend_scanning_pipe = TrendScanning(
#     #     close_name, ts_look_forward_window, ts_min_sample_length, ts_step
#     #     )
#     # ts_fit = trend_scanning_pipe.fit(df)
#     # X = trend_scanning_pipe.transform(df)
# #################### TESTS








# ##########

# def calculate_t_values(subset, min_sample_length, step):  # pragma: no cover
#     """
#     For loop for calculating linear regression every n steps.

#     :param subset: (np.array) subset of indecies for which we want to calculate t values
#     :return: (float) maximum t value and index of maximum t value
#     """
#     max_abs_t_value = -np.inf  # Maximum abs t-value of b_1 coefficient among l values
#     max_t_value_index = None  # Index with maximum t-value

#     for forward_window in np.arange(min_sample_length, subset.shape[0], step):

#         y_subset = subset[:forward_window].reshape(-1, 1)  # y{t}:y_{t+l}

#         # Array of [1, 0], [1, 1], [1, 2], ... [1, l] # b_0, b_1 coefficients
#         x_subset = np.ones((y_subset.shape[0], 2))
#         x_subset[:, 1] = np.arange(y_subset.shape[0])

#         # Get regression coefficients estimates
#         xy_ = x_subset.transpose() @ y_subset
#         xx_ = x_subset.transpose() @ x_subset

#         #   check for singularity
#         det = np.linalg.det(xx_)

#         # get coefficient and std from linear regression
#         if det == 0:
#             b_mean = np.array([[np.nan]])
#             b_std = np.array([[np.nan, np.nan]])
#         else:
#             xx_inv = np.linalg.inv(xx_)
#             b_mean = xx_inv @ xy_
#             err = y_subset - (x_subset @ b_mean)
#             b_std = np.dot(np.transpose(err), err) / (x_subset.shape[0] - x_subset.shape[1]) * xx_inv  # pylint: disable=E1136  # pylint/issues/3139

#         # Check if l gives the maximum t-value among all values {0...L}
#             t_beta_1 = (b_mean[1] / np.sqrt(b_std[1, 1]))[0]
#             if abs(t_beta_1) > max_abs_t_value:
#                 max_abs_t_value = abs(t_beta_1)
#                 max_t_value = t_beta_1
#                 max_t_value_index = forward_window

#     return max_t_value_index, max_t_value


# def trend_scanning_labels(price_series: pd.Series, t_events: list = None, look_forward_window: int = 20,
#                           min_sample_length: int = 5, step: int = 1) -> pd.DataFrame:



# price_series=data['close_orig']
# t_events=cusum_events
# look_forward_window=ts_look_forward_window
# min_sample_length=ts_min_sample_length
# step=ts_step


# # pylint: disable=invalid-name

# if t_events is None:
#     t_events = price_series.index

# t1_array = []  # Array of label end times
# t_values_array = []  # Array of trend t-values

# for index in t_events:
    
#     ###
#     index = t_events[0]
#     ###
    
#     subset = price_series.loc[index:].iloc[:look_forward_window]  # Take t:t+L window
#     if subset.shape[0] >= look_forward_window:

#         # linear regressoin for every index
#         subset, min_sample_length, step = subset.values, min_sample_length, step
        
#         max_abs_t_value = -np.inf  # Maximum abs t-value of b_1 coefficient among l values
#         max_t_value_index = None  # Index with maximum t-value

#         for forward_window in np.arange(min_sample_length, subset.shape[0], step):
            
#             y_subset = subset[:forward_window].reshape(-1, 1)  # y{t}:y_{t+l}

#             # Array of [1, 0], [1, 1], [1, 2], ... [1, l] # b_0, b_1 coefficients
#             x_subset = np.ones((y_subset.shape[0], 2))
#             x_subset[:, 1] = np.arange(y_subset.shape[0])

#             # Get regression coefficients estimates
#             xy_ = x_subset.transpose() @ y_subset
#             xx_ = x_subset.transpose() @ x_subset

#             #   check for singularity
#             det = np.linalg.det(xx_)

#             # get coefficient and std from linear regression
#             if det == 0:
#                 b_mean = np.array([[np.nan]])
#                 b_std = np.array([[np.nan, np.nan]])
#             else:
#                 xx_inv = np.linalg.inv(xx_)
#                 b_mean = xx_inv @ xy_
#                 err = y_subset - (x_subset @ b_mean)
#                 b_std = np.dot(np.transpose(err), err) / (x_subset.shape[0] - x_subset.shape[1]) * xx_inv  # pylint: disable=E1136  # pylint/issues/3139

#                 # Check if l gives the maximum t-value among all values {0...L}
#                 t_beta_1 = (b_mean[1] / np.sqrt(b_std[1, 1]))[0]
#                 if abs(t_beta_1) > max_abs_t_value:
#                     max_abs_t_value = abs(t_beta_1)
#                     max_t_value = t_beta_1
#                     max_t_value_index = forward_window
        
        
        
#         max_t_value_index, max_t_value = calculate_t_values(subset.values,
#                                                             min_sample_length,
#                                                             step)

#         # Store label information (t1, return)
#         label_endtime_index = subset.index[max_t_value_index - 1]
#         t1_array.append(label_endtime_index)
#         t_values_array.append(max_t_value)

#     else:
#         t1_array.append(None)
#         t_values_array.append(None)

# labels = pd.DataFrame({'t1': t1_array, 't_value': t_values_array}, index=t_events)
# labels.loc[:, 'ret'] = price_series.reindex(labels.t1).values / price_series.reindex(labels.index).values - 1
# labels['bin'] = np.sign(labels.t_value)

# return labels



# from mlfinlab.structural_breaks.sadf import get_betas
# # pylint: disable=invalid-name

# if t_events is None:
#     t_events = price_series.index

# t1_array = []  # Array of label end times
# t_values_array = []  # Array of trend t-values

# for index in t_events:
    
#     ####
#     index = t_events[0]
#     ####
    
#     subset = price_series.loc[index:].iloc[:look_forward_window]  # Take t:t+L window
#     if subset.shape[0] >= look_forward_window:
#         # Loop over possible look-ahead windows to get the one which yields maximum t values for b_1 regression coef
#         max_abs_t_value = -np.inf  # Maximum abs t-value of b_1 coefficient among l values
#         max_t_value_index = None  # Index with maximum t-value
#         max_t_value = None  # Maximum t-value signed

#         # Get optimal label end time value based on regression t-statistics
#         for forward_window in np.arange(min_sample_length, subset.shape[0], step):
            
#             y_subset = subset.iloc[:forward_window].values.reshape(-1, 1)  # y{t}:y_{t+l}

#             # Array of [1, 0], [1, 1], [1, 2], ... [1, l] # b_0, b_1 coefficients
#             X_subset = np.ones((y_subset.shape[0], 2))
#             X_subset[:, 1] = np.arange(y_subset.shape[0])

#             # Get regression coefficients estimates
            
            
#             # X=X_subset
#             # y=y_subset

#             # xy = np.dot(X.T, y)
#             # xx = np.dot(X.T, X)

#             # try:
#             #     xx_inv = np.linalg.inv(xx)
#             # except np.linalg.LinAlgError:
#             #     return [np.nan], [[np.nan, np.nan]]

#             # b_mean = np.dot(xx_inv, xy)
#             # err = y - np.dot(X, b_mean)
#             # b_var = np.dot(err.T, err) / (X.shape[0] - X.shape[1]) * xx_inv

#             # return b_mean, b_var

            
            
            
            
    
#             b_mean_, b_std_ = get_betas(X_subset, y_subset)
#             # Check if l gives the maximum t-value among all values {0...L}
#             t_beta_1 = (b_mean_[1] / np.sqrt(b_std_[1, 1]))[0]
#             print(t_beta_1)
#             if abs(t_beta_1) > max_abs_t_value:
#                 max_abs_t_value = abs(t_beta_1)
#                 max_t_value = t_beta_1
#                 max_t_value_index = forward_window

#         # Store label information (t1, return)
#         label_endtime_index = subset.index[max_t_value_index - 1]
#         t1_array.append(label_endtime_index)
#         t_values_array.append(max_t_value)

#     else:
#         t1_array.append(None)
#         t_values_array.append(None)

# labels = pd.DataFrame({'t1': t1_array, 't_value': t_values_array}, index=t_events)
# labels.loc[:, 'ret'] = price_series.reindex(labels.t1).values / price_series.reindex(labels.index).values - 1
# labels['bin'] = labels.t_value.apply(np.sign)

# return labels