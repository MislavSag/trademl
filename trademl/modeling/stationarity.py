import pandas as pd
import numpy as np
from numpy.fft import fft, ifft
# from mlfinlab.features.fracdiff import frac_diff_ffd
from statsmodels.tsa.stattools import adfuller
import numba
from numba import njit
# import matplotlib.pyplot as plt
#### PERFORMANCE !!! FROM 
# https://github.com/cottrell/fractional-differentiation-time-series/blob/master/fracdiff/fracdiff.py
# from: http://www.mirzatrokic.ca/FILES/codes/fracdiff.py
# small modification: wrapped 2**np.ceil(...) around int()
# https://github.com/SimonOuellette35/FractionalDiff/blob/master/question2.py


### PARAMETER
_default_thresh = 1e-4


def get_weights(d, size):
    """Expanding window fraction difference weights."""
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

@numba.njit
def get_weights_ffd(d, thres, lim=99999):
    """Fixed width window fraction difference weights.
    Set lim to be large if you want to only stop at thres.
    Set thres to be zero if you want to ignore it.
    """
    w = [1.0]
    k = 1
    for i in range(1, lim):
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def frac_diff_ffd(x, d, thres=_default_thresh, lim=None):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 1
    if lim is None:
        lim = len(x)
    w, out = _frac_diff_ffd(x, d, lim, thres=thres)
    out = np.array(out)
    # print(f'weights is shape {w.shape}')
    return out


# this method was not faster
# def frac_diff_ffd_stride_tricks(x, d, thres=_default_thresh):
#     """d is any positive real"""
#     assert isinstance(x, np.ndarray)
#     w = get_weights_ffd(d, thres, len(x))
#     width = len(w) - 1
#     output = np.empty(len(x))
#     output[:width] = np.nan
#     output[width:] = np.dot(np.lib.stride_tricks.as_strided(x, (len(x) - width, len(w)), (x.itemsize, x.itemsize)), w[:,0])
#     return output


@numba.njit
def _frac_diff_ffd(x, d, lim, thres=_default_thresh):
    """d is any positive real"""
    w = get_weights_ffd(d, thres, lim)
    width = len(w) - 1
    output = []
    for i in range(0, x.shape[0]):
        if i < width:
            output.append(np.nan)
        else:
            output.append(np.dot(w.T, x[i - width: i + 1])[0])
    return w, output


def fast_frac_diff(x, d):
    """expanding window version using fft form"""
    assert isinstance(x, np.ndarray)
    T = len(x)
    np2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
    k = np.arange(1, T)
    b = (1,) + tuple(np.cumprod((k - d - 1) / k))
    z = (0,) * (np2 - T)
    z1 = b + z
    z2 = tuple(x) + z
    dx = ifft(fft(z1) * fft(z2))
    return np.real(dx[0:T])


# TESTS


# def test_all():
#     for d in [0.3, 1, 1.5, 2, 2.5]:
#         test_fast_frac_diff_equals_fracDiff_original_impl(d=d)
#         test_frac_diff_ffd_equals_original_impl(d=d)
#         # test_frac_diff_ffd_equals_prado_original(d=d) # his implementation is busted for fractional d


# def test_frac_diff_ffd_equals_prado_original(d=3):
#     # ignore this one for now as Prado's version does not work
#     from .prado_orig import fracDiff_FFD_prado_original
#     import pandas as pd
# 
#     x = np.random.randn(100)
#     a = frac_diff_ffd(x, d, thres=_default_thresh)
#     b = fracDiff_FFD_prado_original(pd.DataFrame(x), d, thres=_default_thresh)
#     b = np.squeeze(b.values)
#     a = a[d:]  # something wrong with the frac_diff_ffd gives extra entries of zero
#     assert np.allclose(a, b)
#     # return locals()


# def test_frac_diff_ffd_equals_original_impl(d=3):
#     from .prado_orig import fracDiff_FFD_original_impl
#     import pandas as pd

#     x = np.random.randn(100)
#     a = frac_diff_ffd(x, d, thres=_default_thresh)
#     b = fracDiff_FFD_original_impl(pd.DataFrame(x), d, thres=_default_thresh)
#     assert np.allclose(a, b)
#     # return locals()


# def test_fast_frac_diff_equals_fracDiff_original_impl(d=3):
#     from .prado_orig import fracDiff_original_impl
#     import pandas as pd

#     x = np.random.randn(100)
#     a = fast_frac_diff(x, d)
#     b = fracDiff_original_impl(pd.DataFrame(x), d, thres=None)
#     b = b.values
#     assert a.shape == b.shape
#     assert np.allclose(a, b)
#     # return locals()


### MY FUNCTIONS #####

    
def min_ffd_plot(unstationary_pdseries):
    """Plot:
    1) correlation bretween first current value and first lag 
    2) 5% pavlaue for ADF test
    3 ) mean of ADF 95% confidence
    
    Arguments:
        unstationary_pdseries {pd.Series} -- pd.Series you want to plot
    """
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    for d in np.linspace(0, 1, 11):
        df1 = np.log(unstationary_pdseries).resample('1D').last()  # downcast to daily obs        
        df1.dropna(inplace=True)
        df2 = frac_diff_ffd(df1.values, d=d, thres=1e-4).dropna()
        corr = np.corrcoef(df1.loc[df2.index].squeeze(), df2.squeeze())[0, 1]
        df2 = adfuller(df2.squeeze(), maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr]  # with critical value
    out[['adfStat', 'corr']].plot(secondary_y='adfStat', figsize=(10, 8))
    plt.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    return


def min_ffd_value(unstationary_series, d_domain, pvalue_threshold=0.05):
    """
    Source: Chapter 5, AFML (section 5.5, page 83);
    Minimal value of d which makes pandas series stationary.
    References:
    https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
    https://wwwf.imperial.ac.uk/~ejm/M3S8/Problems/hosking81.pdf
    Constant width window (new solution)
    Note 1: thresh determines the cut-off weight for the window
    Note 2: diff_amt can be any positive fractional, not necessarity bounded [0, 1].
    :param unstationary_series: (pd.Series)
    :param d_domain: (np.array) numpy linspace; possible d values
    :param pvalue_threshold: (float) ADF p-value threshold above which nonstationary
    :return: (float) minimum value of d which makes series stationary
    """
    d_min = None
    for d_i in d_domain:
        
        # resaample series to daily frequency
        df1 = unstationary_series.resample('1D').last()
        df1.dropna(inplace=True)
        df1 = df1.squeeze()
        
        # fracDiff for d
        df2 = frac_diff_ffd(df1.values, d=d_i, thres=1e-4, lim=None)
        df2 = pd.Series(df2, index=df1.index).dropna()

        # ADF test
        df2 = adfuller(df2.squeeze(), maxlag=1, regression='c', autolag=None)

        # if p-value is grater than threshold stop and return d
        if df2[1] <= pvalue_threshold:
            d_min = d_i
            break

    return d_min


def min_ffd_all_cols(data):
    """
    Get min_d for all columns
    
    :param data: (pd.DataFrame) Pandas DF with unstationary columns.
    :return: (pd.DataFrame) Pandas DF with stationary columns.
    """
    # stationarity tests
    adfTest = data.apply(lambda x: adfuller(x, 
                                            maxlag=1,
                                            regression='c',
                                            autolag=None),
                         axis=0)
    stationaryCols = adfTest.columns[adfTest.iloc[1] > 0.1]
    # adfTestPval = [adf[1] for adf in adfTest]
    # adfTestPval = pd.Series(adfTestPval)
    # stationaryCols = data.loc[:, (adfTestPval > 0.1).to_list()].columns

    # get minimum values of d for every column
    seq = np.linspace(0, 1, 16)
    min_d = data[stationaryCols].apply(lambda x: min_ffd_value(x.to_frame(), seq))
    
    return stationaryCols, min_d


def unstat_cols_to_stat(data, min_d, stationaryCols):
    """
    Convert unstationary columns to stationary.
    
    :param data: (pd.DataFrame) Pandas DF with unstationary columns.
    :return: (pd.DataFrame) Pandas DF with stationary columns.
    """
    # # stationarity tests
    # adfTest = data.apply(lambda x: adfuller(x, maxlag=1, regression='c',
    #                                     autolag=None), axis=0)
    # adfTestPval = [adf[1] for adf in adfTest]
    # adfTestPval = pd.Series(adfTestPval)
    # stationaryCols = data.loc[:, (adfTestPval > 0.1).to_list()].columns

    # # get minimum values of d for every column
    # seq = np.linspace(0, 1, 16)
    # min_d = data[stationaryCols].apply(lambda x: min_ffd_value(x.to_frame(), seq))

    # make stationary spy
    dataStationary = data[stationaryCols].loc[:, min_d > 0]
    diff_amt_args = min_d[min_d > 0].to_list()
    for i, col in enumerate(dataStationary.columns):
        print("Making ", col, " stationary")
        dataStationary[col] = frac_diff_ffd(dataStationary[col].values, diff_amt_args[i])

    # add stationry spy to spy
    columnsToChange = data[stationaryCols].loc[:, min_d > 0].columns
    data[columnsToChange] = dataStationary
    data.dropna(inplace=True)

    return data
