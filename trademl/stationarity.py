import pandas as pd
import numpy as np
from mlfinlab.features.fracdiff import frac_diff_ffd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


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
        df2 = frac_diff_ffd(df1, diff_amt=d, thresh=0.01).dropna()
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

    for d_i in d_domain:
        # resaample series to daily frequency
        df1 = np.log(unstationary_series).resample('1D').last()
        df1.dropna(inplace=True)

        # fracDiff for d
        df2 = frac_diff_ffd(df1, diff_amt=d_i, thresh=0.01).dropna()

        # ADF test
        df2 = adfuller(df2.squeeze(), maxlag=1, regression='c', autolag=None)

        # if p-value is grater than threshold stop and return d
        if df2[1] <= pvalue_threshold:
            d_min = d_i
            break

    return d_min


def unstat_cols_to_stat(data):
    """
    Convert unstationary columns to stationary.
    
    :param data: (pd.DataFrame) Pandas DF with unstationary columns.
    :return: (pd.DataFrame) Pandas DF with stationary columns.
    """
    # stationarity tests
    adfTest = data.apply(lambda x: adfuller(x, maxlag=1, regression='c',
                                            autolag=None),
                         axis=0)
    adfTestPval = [adf[1] for adf in adfTest]
    adfTestPval = pd.Series(adfTestPval)
    stationaryCols = data.loc[:, (adfTestPval > 0.1).to_list()].columns

    # get minimum values of d for every column
    seq = np.linspace(0, 1, 16)
    min_d = data[stationaryCols].apply(lambda x: min_ffd_value(x.to_frame(), seq))

    # make stationary spy
    spyStationary = data[stationaryCols].loc[:, min_d > 0]
    diff_amt_args = min_d[min_d > 0].to_list()
    for i, col in enumerate(spyStationary.columns):
        print("Making ", col, " stationary")
        spyStationary[col] = frac_diff_ffd(spyStationary[[col]], diff_amt_args[i])

    # add stationry spy to spy
    columnsToChange = data[stationaryCols].loc[:, min_d > 0].columns
    data[columnsToChange] = spyStationary
    data.dropna(inplace=True)
    
    return data
