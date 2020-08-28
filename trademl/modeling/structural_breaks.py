'''
STRUCTURAL BREAKS
'''

import pandas as pd
import numpy as np
from numba import njit, prange
import mlfinlab as ml


# Chow-Type Dickey-Fuller Test
@njit
def _get_dfc_for_t(molecule_range, series_lag_values_start, series_diff, series_lag_values):
    """
    Get Chow-Type Dickey-Fuller Test statistics for each index in molecule
    :param molecule_range: (np.array) of dates to test
    :param series_lag_values_start: (int) offset series because of min_length
    :return: (pd.Series) fo statistics for each index from molecule
    """
    dfc_series = []
    for i in molecule_range:
        ### TEST
        # index = molecule[0]
        ### TEST
        series_lag_values_ = series_lag_values.copy()
        series_lag_values_[:(series_lag_values_start + i)] = 0  # D_t* indicator: before t* D_t* = 0

        # define x and y for regression
        y = series_diff
        x = series_lag_values_.reshape(-1, 1)
        
        # Get regression coefficients estimates
        xy = x.transpose() @ y
        xx = x.transpose() @ x

        # calculate to check for singularity
        det = np.linalg.det(xx)

        # get coefficient and std from linear regression
        if det == 0:
            b_mean = [np.nan]
            b_std = [[np.nan, np.nan]]
        else:
            xx_inv = np.linalg.inv(xx)
            coefs = xx_inv @ xy
            err = y - (x @ coefs)
            coef_vars = np.dot(np.transpose(err), err) / (x.shape[0] - x.shape[1]) * xx_inv
            
        b_estimate, b_var = coefs[0], coef_vars[0][0]
        dfc_series.append(b_estimate / (b_var ** 0.5))
        
    return dfc_series


def get_chow_type_stat(series: pd.Series, min_length: int = 20) -> pd.Series:
    """
    Multithread implementation of Chow-Type Dickey-Fuller Test, p.251-252
    :param series: (pd.Series) series to test
    :param min_length: (int) minimum sample length used to estimate statistics
    :param num_threads: (int): number of cores to use
    :return: (pd.Series) of Chow-Type Dickey-Fuller Test statistics
    """
    # Indices to test. We drop min_length first and last values
    molecule = series.index[min_length:series.shape[0] - min_length]
    molecule = molecule.values
    molecule_range = np.arange(0, len(molecule))

    series_diff = series.diff().dropna()
    series_diff = series_diff.values
    series_lag = series.shift(1).dropna()
    series_lag_values = series_lag.values
    series_lag_times_ = series_lag.index.values
    series_lag_values_start = np.where(series_lag_times_ == molecule[0])[0].item() + 1
    
    dfc_series = _get_dfc_for_t(molecule_range, series_lag_values_start, series_diff, series_lag_values)
    
    dfc_series = pd.Series(dfc_series, index=molecule)
    
    return dfc_series



### SADF
from typing import Union, Tuple


def _lag_df(df: pd.DataFrame, lags: Union[int, list]) -> pd.DataFrame:
    """
    Advances in Financial Machine Learning, Snipet 17.3, page 259.
    Apply Lags to DataFrame
    :param df: (int or list) Either number of lags to use or array of specified lags
    :param lags: (int or list) Lag(s) to use
    :return: (pd.DataFrame) Dataframe with lags
    """
    df_lagged = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(1, lags + 1)
    else:
        lags = [int(lag) for lag in lags]

    for lag in lags:
        temp_df = df.shift(lag).copy(deep=True)
        temp_df.columns = [str(i) + '_' + str(lag) for i in temp_df.columns]
        df_lagged = df_lagged.join(temp_df, how='outer')
    return df_lagged

def _get_y_x(series: pd.Series, model: str, lags: Union[int, list],
             add_const: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Advances in Financial Machine Learning, Snippet 17.2, page 258-259.
    Preparing The Datasets
    :param series: (pd.Series) Series to prepare for test statistics generation (for example log prices)
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param lags: (int or list) Either number of lags to use or array of specified lags
    :param add_const: (bool) Flag to add constant
    :return: (pd.DataFrame, pd.DataFrame) Prepared y and X for SADF generation
    """
    series = pd.DataFrame(series)
    series_diff = series.diff().dropna()
    x = _lag_df(series_diff, lags).dropna()
    x['y_lagged'] = series.shift(1).loc[x.index]  # add y_(t-1) column
    y = series_diff.loc[x.index]

    if add_const is True:
        x['const'] = 1

    if model == 'linear':
        x['trend'] = np.arange(x.shape[0])  # Add t to the model (0, 1, 2, 3, 4, 5, .... t)
        beta_column = 'y_lagged'  # Column which is used to estimate test beta statistics
    elif model == 'quadratic':
        x['trend'] = np.arange(x.shape[0]) # Add t to the model (0, 1, 2, 3, 4, 5, .... t)
        x['quad_trend'] = np.arange(x.shape[0]) ** 2 # Add t^2 to the model (0, 1, 4, 9, ....)
        beta_column = 'y_lagged'  # Column which is used to estimate test beta statistics
    elif model == 'sm_poly_1':
        y = series.loc[y.index]
        x = pd.DataFrame(index=y.index)
        x['const'] = 1
        x['trend'] = np.arange(x.shape[0])
        x['quad_trend'] = np.arange(x.shape[0]) ** 2
        beta_column = 'quad_trend'
    elif model == 'sm_poly_2':
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x['const'] = 1
        x['trend'] = np.arange(x.shape[0])
        x['quad_trend'] = np.arange(x.shape[0]) ** 2
        beta_column = 'quad_trend'
    elif model == 'sm_exp':
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x['const'] = 1
        x['trend'] = np.arange(x.shape[0])
        beta_column = 'trend'
    elif model == 'sm_power':
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x['const'] = 1
        # TODO: Rewrite logic of this module to avoid division by zero
        with np.errstate(divide='ignore'):
            x['log_trend'] = np.log(np.arange(x.shape[0]))
        beta_column = 'log_trend'
    else:
        raise ValueError('Unknown model')

    # Move y_lagged column to the front for further extraction
    columns = list(x.columns)
    columns.insert(0, columns.pop(columns.index(beta_column)))
    x = x[columns]
    return x, y


@numba.njit
def get_betas(X: np.array, y: np.array) -> Tuple[np.array, np.array]:
    """
    Advances in Financial Machine Learning, Snippet 17.4, page 259.
    Fitting The ADF Specification (get beta estimate and estimate variance)
    :param X: (pd.DataFrame) Features(factors)
    :param y: (pd.DataFrame) Outcomes
    :return: (np.array, np.array) Betas and variances of estimates
    """

    # Get regression coefficients estimates
    xy_ = np.dot(X.T, y)
    xx_ = np.dot(X.T, X)

    #   check for singularity
    det = np.linalg.det(xx_)

    # get coefficient and std from linear regression
    if det == 0:
        b_mean = np.array([[np.nan]])
        b_var = np.array([[np.nan, np.nan]])
        return None
    else:
        xx_inv = np.linalg.inv(xx_)
        b_mean = np.dot(xx_inv, xy_)
        err = y - np.dot(X, b_mean)
        b_var = np.dot(np.transpose(err), err) / (X.shape[0] - X.shape[1]) * xx_inv  # pylint: disable=E1136  # pylint/issues/3139
        return b_mean, b_var



@numba.njit
def _get_sadf_at_t(X: pd.DataFrame, y: pd.DataFrame, min_length: int, model: str, phi: float) -> float:
    """
    Advances in Financial Machine Learning, Snippet 17.2, page 258.
    SADF's Inner Loop (get SADF value at t)
    :param X: (pd.DataFrame) Lagged values, constants, trend coefficients
    :param y: (pd.DataFrame) Y values (either y or y.diff())
    :param min_length: (int) Minimum number of samples needed for estimation
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :return: (float) SADF statistics for y.index[-1]
    """
    start_points = range(0, y.shape[0] - min_length + 1)
    bsadf = -np.inf
    for start in start_points:
        y_, X_ = y[start:], X[start:]
        b_mean_, b_std_ = get_betas(X_, y_)
        # if b_mean_ is not None:  DOESNT WORK WITH NUMBA
        b_mean_, b_std_ = b_mean_[0, 0], b_std_[0, 0] ** 0.5
        # Rewrite logic of this module to avoid division by zero
        if b_std_ != np.float64(0):
            all_adf = b_mean_ / b_std_
        if model[:2] == 'sm':
            all_adf = np.abs(all_adf) / (y.shape[0]**phi)
        if all_adf > bsadf:
            bsadf = all_adf
    return bsadf


@numba.njit
def _sadf_outer_loop(X: np.array, y: np.array, min_length: int, model: str, phi: float,
                     ) -> pd.Series:
    """
    This function gets SADF for t times from molecule
    :param X: (pd.DataFrame) Features(factors)
    :param y: (pd.DataFrame) Outcomes
    :param min_length: (int) Minimum number of observations
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :param molecule: (list) Indices to get SADF
    :return: (pd.Series) SADF statistics
    """
    sadf_series_val = []
    for index in range(1, (y.shape[0]-min_length+1)):
        X_subset = X[:min_length+index]
        y_subset = y[:min_length+index]
        value = _get_sadf_at_t(X_subset, y_subset, min_length, model, phi)
        sadf_series_val.append(value)
    return sadf_series_val



def my_get_sadf(series: pd.Series, model: str, lags: Union[int, list], min_length: int, add_const: bool = False,
             phi: float = 0, num_threads: int = 8, verbose: bool = True) -> pd.Series:
    
    X, y = _get_y_x(series, model, lags, add_const)
    molecule = y.index[min_length:y.shape[0]]
    X_val = X.values
    y_val = y.values
    
    sadf_series = _sadf_outer_loop(X=X.values, y=y.values,
                                   min_length=min_length, model=model, phi=phi)
    sadf_series_val = np.array(sadf_series)
    
    return sadf_series_val
