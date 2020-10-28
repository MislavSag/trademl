import pandas as pd
import numpy as np
import mlfinlab.microstructural_features as micro
import trademl as tml
from talib.abstract import (
    DEMA, EMA, MIDPRICE, SMA, T3, TEMA, TRIMA, WMA,
    ADX, ADXR, AROONOSC, BOP, CMO, DX, MFI, MINUS_DM, MOM, ROC, RSI,
    TRIX , WILLR, ATR, NATR, BBANDS, AROON, STOCHRSI,
    HT_TRENDLINE, AD, OBV, HT_DCPERIOD, HT_DCPHASE, HT_TRENDMODE,
    TRANGE, AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE, ULTOSC,
    MAMA, SAR, SAREXT, APO, MACD, ADOSC,
    HT_PHASOR, HT_SINE, STOCHF, STOCH,
    BETA, CORREL, LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE, TSF
)
from sklearn.base import BaseEstimator, TransformerMixin
from trademl.modeling.utils import time_method
from gplearn.genetic import SymbolicTransformer
from gplearn.functions import make_function
from scipy.signal import savgol_filter


def add_ind(ohlcv, f, n, periods):
    """
    Add technical indicator to pd.DataFrame

    Parameters
    ----------
    f : function
        function from ta_lib package.
    n : str
        Nme prefix.

    Returns
    -------
    pd.Data.Frame.

    """
    ind = pd.concat([f(ohlcv, p).rename(n + str(p)) for p in periods],
                    axis=1)
    return ind


def add_ind_df(ohlcv, f, n, periods):
    """
    Add technical indicator to pd.DataFrame when indicator has multiplie
    outputs.

    Parameters
    ----------
    f : function
        function from ta_lib package.
    n : str
        Nme prefix.

    Returns
    -------
    pd.Data.Frame.

    """
    ind = [f(ohlcv, p).add_prefix((f._Function__namestr + '_' + str(p) + '_'))  
           for p in periods]
    ind = pd.concat(ind, axis=1)
    return ind


@time_method
def add_technical_indicators(data, periods):
    """Add tecnical indicators as featues.
    
    Arguments:
        data {pd.DataFrame} -- Pandas data frame with OHLC data
        periods {list} -- List that contain periods as arguments.
    
    Returns:
        pd.dataFrame -- Pandas data frame with additional indicators
    """
    # add technical indicators for variuos periods when ind has 1 output
    indsList = [DEMA, EMA, MIDPRICE, SMA, T3, # MIDPOINT
                TEMA, TRIMA, WMA,  # KAMA memory intensive!
                ADX, ADXR, AROONOSC, BOP, CMO, DX, MFI, MINUS_DM, MOM, ROC, RSI,
                TRIX , WILLR,  # CCI DOEANST WORK?
                ATR, NATR,
                BETA, CORREL, LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT,
                LINEARREG_SLOPE, TSF]  # OVDJE NASTAVITI S NIZOM!!!
    inds = [add_ind(data, f, f._Function__name.decode('ascii'), periods)
            for f in indsList]
    inds = pd.concat(inds, axis=1)
    data = pd.concat([data, inds], axis=1)

    #  add technical indicators for variuos periods when ind has multiplie
    # outputs
    indsList = [BBANDS, AROON, STOCHRSI]
    inds = [add_ind_df(data, f, f._Function__name.decode('ascii'), periods)
            for f in indsList]
    inds = pd.concat(inds, axis=1)
    data = pd.concat([data, inds], axis=1)

    # add tecnical indicators with no function arguments
    indsList = [HT_TRENDLINE, AD, OBV, HT_DCPERIOD, HT_DCPHASE, HT_TRENDMODE,
                TRANGE, AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE, 
                ULTOSC]
    inds = [f(data).rename(f._Function__name.decode('ascii')) for f in indsList]
    inds = pd.concat(inds, axis=1)
    data = pd.concat([data, inds], axis=1)

    # add other indicators    
    data[['MAMA', 'FAMA']] = MAMA(data)  # MAVP ne radi
    data[['MAMA_25', 'FAMA_25']] = MAMA(data, fastlimit=0.25, slowlimit=0.02)  # MAVP ne radi
    data[['MAMA_5', 'FAMA_5']] = MAMA(data, fastlimit=0.5, slowlimit=0.05)  # MAVP ne radi
    data['SAR'] = SAR(data)
    data['SAR_1'] = SAR(data, acceleration=0.01, maximum=0.01)
    data['SAR_2'] = SAR(data, acceleration=0.02, maximum=0.02)
    data['SAREXT'] = SAREXT(data)
    startvalue, offsetonreverse, accelerationinitlong, accelerationlong,\
    accelerationmaxlong, accelerationinitshort, accelerationshort,\
    accelerationmaxshort = np.random.uniform(low=0.01, high=0.4, size=8)
    data['SAREXT_rand'] = SAREXT(data, startvalue=startvalue, 
                                 offsetonreverse=offsetonreverse,
                                 accelerationinitlong=accelerationinitlong,
                                 accelerationlong=accelerationlong,
                                 accelerationmaxlong=accelerationmaxlong,
                                 accelerationinitshort=accelerationinitshort,
                                 accelerationshort=accelerationshort,
                                 accelerationmaxshort=accelerationmaxshort)
    data['APO'] = APO(data)
    data['APO_1'] = APO(data, fastperiod=24, slowperiod=52, matype=0)
    data['APO_2'] = APO(data, fastperiod=50, slowperiod=100, matype=0)
    data['APO_3'] = APO(data, fastperiod=100, slowperiod=200, matype=0)
    data['APO_4'] = APO(data, fastperiod=200, slowperiod=400, matype=0)
    data['APO_5'] = APO(data, fastperiod=12000, slowperiod=24000, matype=0)
    data['ADOSC'] = ADOSC(data)
    data[['MACD', 'MACDSIGNAL', 'MACDHIST']] = MACD (data)
    data[['MACD_24', 'MACDSIGNAL_24', 'MACDHIST_24']] = MACD (data,
                                                              fastperiod=24,
                                                              slowperiod=52,
                                                              signalperiod=18)
    data[['MACD_48', 'MACDSIGNAL_48', 'MACDHIST_48']] = MACD (data,
                                                            fastperiod=48,
                                                            slowperiod=104,
                                                            signalperiod=36)
    data[['MACD_200', 'MACDSIGNAL_200', 'MACDHIST_200']] = MACD (data,
                                                        fastperiod=200,
                                                        slowperiod=300,
                                                        signalperiod=50)
    # data[['MACDFIX', 'MACDFIX SIGNAL', 'MACDFIXHIST']] = MACDFIX(data)
    # data[['MACDFIX_18', 'MACDFIX SIGNAL_18',
    #       'MACDFIXHIST_18']] = MACDFIX(data, 18)
    # data[['MACDFIX_50', 'MACDFIX SIGNAL_50',
    #       'MACDFIXHIST_50']] = MACDFIX(data, 50)
    # data[['MACDFIX_200', 'MACDFIX SIGNAL_200',
    #       'MACDFIXHIST_200']] = MACDFIX(data, 200)
    # data[['MACDFIX_12000', 'MACDFIX SIGNAL_12000',
    #       'MACDFIXHIST_12000']] = MACDFIX(data, 12000)
    data[['inphase', 'quadrature']] = HT_PHASOR(data)
    data[['sine', 'leadsine']] = HT_SINE(data)
    data[['fastk', 'fastd']]= STOCHF(data)
    data[['fastk_20', 'fastd_20']]= STOCHF(data, fastk_period=20, fastd_period=9, fastd_matype=0)
    data[['fastk_200', 'fastd_200']]= STOCHF(data, fastk_period=200, fastd_period=80, fastd_matype=0)
    data[['fastk_3600', 'fastd_3600']]= STOCHF(data, fastk_period=3600, fastd_period=400, fastd_matype=0)
    data[['slowk', 'slowd']]= STOCH(data)
    data[['slowk_30', 'slowd_30']]= STOCH(data, fastk_period=30, slowk_period=15,
                                          slowk_matype=0, slowd_period=9, slowd_matype=0)
        
    return data


def add_fourier_transform(data, col, periods):
    """
    Calculate Fourier transformation of time series for for given periods.

    Arguments:
        data {pd.DataFrame} -- Pandas data frame
        col {str} -- Column you want to transform.
        periods {list} -- List that contain periods as arguments.
    
    Returns:
        [pd.DataFrame] -- Pandas DataFrame with new transformed columns.
    """
    close_fft = np.fft.fft(np.asarray(data[col].tolist()))
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in periods:
        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[num_:-num_] = 0
        data['fft_' + str(num_)] = np.abs(fft_list_m10)
    
    return data


def range_grow(start=5, steps=9, pct=.7):
    s = [start]
    for i in range(0, steps):
        s.append(s[i] + s[i] * pct)
    return [round(x) for x in s]


def add_ohlcv_features(data):
    """
    Calculate features based on OHLCV data and add tham to data feame.
    """

    # add ohlc transformations
    data['high_low'] = data['high'] - data['low']
    data['close_open'] = data['close'] - data['open']
    data['close_ath'] = data['close'].cummax()
        
    # simple momentum
    data['momentum1'] = data['close'].pct_change(periods=1)
    data['momentum2'] = data['close'].pct_change(periods=2)
    data['momentum3'] = data['close'].pct_change(periods=3)
    data['momentum4'] = data['close'].pct_change(periods=4)
    data['momentum5'] = data['close'].pct_change(periods=5)
    data['momentum10'] = data['close'].pct_change(periods=10)
    
    # Volatility
    data['volatility_60'] = np.log(data['close']).diff().rolling(
        window=60, min_periods=60, center=False).std()
    data['volatility_30'] = np.log(data['close']).diff().rolling(
        window=30, min_periods=30, center=False).std()
    data['volatility_15'] = np.log(data['close']).diff().rolling(
        window=15, min_periods=15, center=False).std()
    data['volatility_10'] = np.log(data['close']).diff().rolling(
        window=10, min_periods=10, center=False).std()
    data['volatility_5'] =np.log(data['close']).diff().rolling(
        window=5, min_periods=5, center=False).std()
    
    # Skewness
    data['skew_60'] = np.log(data['close']).diff().rolling(
        window=60, min_periods=60, center=False).skew()
    data['skew_30'] = np.log(data['close']).diff().rolling(
        window=30, min_periods=30, center=False).skew()
    data['skew_15'] = np.log(data['close']).diff().rolling(
        window=15, min_periods=15, center=False).skew()
    data['skew_10'] = np.log(data['close']).diff().rolling(
        window=10, min_periods=10, center=False).skew()
    data['skew_5'] =np.log(data['close']).diff().rolling(
        window=5, min_periods=5, center=False).skew()

    # kurtosis
    data['kurtosis_60'] = np.log(data['close']).diff().rolling(
        window=60, min_periods=60, center=False).kurt()
    data['kurtosis_30'] = np.log(data['close']).diff().rolling(
        window=30, min_periods=30, center=False).kurt()
    data['kurtosis_15'] = np.log(data['close']).diff().rolling(
        window=15, min_periods=15, center=False).kurt()
    data['kurtosis_10'] = np.log(data['close']).diff().rolling(
        window=10, min_periods=10, center=False).kurt()
    data['kurtosis_5'] =np.log(data['close']).diff().rolling(
        window=5, min_periods=5, center=False).kurt()
    
    # microstructural features
    data['roll_measure'] = micro.get_roll_measure(data['close'])
    data['corwin_schultz_est'] = micro.get_corwin_schultz_estimator(
        data['high'], data['low'], 100)
    data['bekker_parkinson_vol'] = micro.get_bekker_parkinson_vol(
        data['high'], data['low'], 100)
    data['kyle_lambda'] = micro.get_bekker_parkinson_vol(
        data['close'], data['volume'])
    data['amihud_lambda'] = micro.get_bar_based_amihud_lambda(
        data['close'], data['volume'])
    data['hasbrouck_lambda'] = micro.get_bar_based_hasbrouck_lambda(
        data['close'], data['volume'])
    tick_diff = data['close'].diff()
    data['tick_rule'] = np.where(tick_diff != 0,
                                np.sign(tick_diff),
                                np.sign(tick_diff).shift(periods=-1))
    
    # Add time features
    data['minute'] = data.index.minute
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['week_of_month'] = data.index.to_series().apply(lambda d: (d.day - 1) // 7 + 1)
    
    # Add smoothed lags
    smooth_name = f"smooth_close"
    data[smooth_name] = savgol_filter(data['close'], 31, 3)
    data = data.assign(**{
        f"{smooth_name}_lag_{t}": data[[smooth_name]].shift(t)
        for t in list(dict.fromkeys(range_grow(1, 150, .055)))
    })
    
    ### ADD VIX TO DATABASE
    q = 'SELECT date, open AS open_vix, high AS high_vix, low AS low_vix, \
        close AS close_vix, volume AS volume_vix FROM VIX'
    data_vix = tml.modeling.utils.query_to_db(q, 'odvjet12_market_data_usa')
    data_vix.set_index(data_vix.date, inplace=True)
    data_vix.drop(columns=['date'], inplace=True)
    data_vix.sort_index(inplace=True)
    # merge spy and vix with merge_asof which uses nearest back value for NA
    data_vix = data_vix.sort_index()
    data = pd.merge_asof(data, data_vix, left_index=True, right_index=True)
    
    ### VIX FEATURES
    data['vix_high_low'] = data['high'] - data['low']
    data['vix_close_open'] = data['close'] - data['open']

    
    return data


def exponent(x):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 100, np.exp(x), 0.)


class Genetic(BaseEstimator, TransformerMixin):

    def __init__(self, population=50000, generations=10, hall_of_fame=500, components=200, metric='spearman'):
        self.state = {}
        self.population = population
        self.generations = generations
        self.hall_of_fame = hall_of_fame
        self.components = components
        self.metric = metric

        # population: Number of formulas per generation
        # generations: Number of generations
        # hall_of_fame: Best final evolution program to evaluate
        # components: X least correlated from the hall of fame
        # metric: pearson for linear model, spearman for tree based estimators

    def fit(self, X, y=None, state={}):
        exponential = make_function(function=exponent, name='exp', arity=1)

        function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max',
                        'min', 'tan', 'sin', 'cos', exponential]

        gp = SymbolicTransformer(generations=self.generations, population_size=self.population,
                                 hall_of_fame=self.hall_of_fame, n_components=self.components,
                                 function_set=function_set,
                                 parsimony_coefficient='auto',
                                 max_samples=0.6, verbose=1, metric=self.metric,
                                 random_state=0, n_jobs=7)

        self.state['genetic'] = {}
        self.state['genetic']['fit'] = gp.fit(X, y)

        return self

    def transform(self, X, y=None, state={}):
        features = self.state['genetic']['fit'].transform(X)
        features = pd.DataFrame(features, columns=["genetic_" + str(a) for a in range(features.shape[1])], index=X.index)
        X = X.join(features)

        return X, y, self.state


class AddFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, add_ta=True, ta_periods=[10, 100]):
        self.add_ta = add_ta
        self.ta_periods = ta_periods

    def fit(self, X, y=None):
        print('Adding features')
        
        return self
    
    @time_method
    def transform(self, X, y=None):
        
        # add tecnical indicators
        if self.add_ta:
            X = add_technical_indicators(X, periods=self.ta_periods)
            X.columns = [cl[0] if isinstance(cl, tuple) else cl for cl in X.columns]
            print('Technical indicators added')
        
        # add other features
        X = add_ohlcv_features(X)
        print('Microstructural and other features added')
        
        # remove na
        if self.add_ta:
            X = X.loc[:, X.isna().sum() < (max(self.ta_periods) + 10)]
        cols_remove_na = range((np.where(X.columns == 'volume')[0].item() + 1), X.shape[1])
        X.dropna(subset=X.columns[cols_remove_na], inplace=True)
        
        return X


###### WEASEL #######
# X_train_sample = X_train.iloc[:1000]
# y_train_sample = y_train.iloc[:1000]
# weasel = WEASEL(sparse=False)
# weasel.fit(X_train_sample, y_train_sample)
# weasel_feature = weasel.transform(X_train)
# weasel_feature.shape
# X_train_sample.shape


# from pyts.datasets import load_gunpoint
# from pyts.transformation import WEASEL
# from sklearn.decomposition import TruncatedSVD

# X_train_pyts, X_test_pyts, y_train_pyts, __pyts = load_gunpoint(return_X_y=True)

# X_train.shape

# weasel = WEASEL(sparse=False)
# weasel.fit(X_train, y_train)
# WEASEL(...)
# >>>len(weasel.vocabulary_)

# >>> weasel.transform(X_test).shape

# close_sequence = data['close']
# def close_to_3d(data, cusum_events, time_step_length):
#     cusum_events_ = cusum_events.intersection(data.index)
#     lstm_sequences = []
#     targets = []
#     for date in cusum_events_:
#         observation = data[:date].iloc[-time_step_length:]
#         if observation.shape[0] < time_step_length or data.index[-1] < date:
#             next
#         else:
#             lstm_sequences.append(observation.values.reshape((1, observation.shape[0])))
#             # targets.append(target_vec[target_vec.index == date])
#     lstm_sequences_all = np.vstack(lstm_sequences)
#     # targets = np.vstack(targets)
#     # targets = targets.astype(np.int64)
#     return lstm_sequences_all


# close_sequence = data['close'].loc[:X_train.index[-1]]
# close_3d_test = close_to_3d(close_sequence, labeling_info.index, time_step_length=10)
# close_3d_test.shape
# y_train.shape
# y_train_seq = y_train_seq.reshape(-1)

# window_sizes = np.ceil(np.array([.1, .3, .5, .7, .9]) * time_step_length).astype(np.int64)
# window_sizes = window_sizes[window_sizes > 5]
# weasel = WEASEL(word_size=5, n_bins=5, window_sizes=window_sizes, sparse=False)
# weasel.fit(close_3d_test, y_train)
# features = weasel.transform(close_3d_test)
# tsvd = TruncatedSVD(n_components=10)  # Reduce sparce matrix
# weasel_tsvd = tsvd.fit(features)
# weasel_tsvd = tsvd.transform(features)
# features = pd.DataFrame(features, columns=["weasel_" + str(a) for a in range(features.shape[1])],
#                         index=X_train.index)
# features.iloc[:, 15].value_counts()