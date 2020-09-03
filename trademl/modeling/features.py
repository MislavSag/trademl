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
    # ind = [f(ohlcv, p).
    #        set_axis((f._Function__namestr + '_' +
    #                  pd.Series(f.output_names) + '_' + str(p)), axis=1)
    #        for p in periods]
    ind = pd.concat(ind, axis=1)
    return ind


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
                TRIX , WILLR,  # CCI NE RADI (VALJDA)
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
