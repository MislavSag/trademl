import glob
import os
import numpy as np
import pandas as pd
from numba import njit, prange
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mlfinlab.structural_breaks import (
    get_chu_stinchcombe_white_statistics,
    get_chow_type_stat, get_sadf)
import mlfinlab as ml
import mlfinlab.microstructural_features as micro
import trademl as tml

from trademl.modeling.utils import time_method



### PANDAS OPTIONS
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


### IMPORT DATA
# import data from mysql database and 
contract = 'SPY'
q = 'SELECT date, open, high, low, close, volume FROM SPY'
data = tml.modeling.utils.query_to_db(q, 'odvjet12_market_data_usa')
data.set_index(data.date, inplace=True)
data.drop(columns=['date'], inplace=True)
data.sort_index(inplace=True)


### REMOVE OUTLIERS
print(data.shape)
data = tml.modeling.outliers.remove_ourlier_diff_median(data, 25)
print(data.shape)
    

# NON SPY OLD WAY
# paths = glob.glob(DATA_PATH + 'ohlcv/*')
# contracts = [os.path.basename(p).replace('.h5', '') for p in paths]
# with pd.HDFStore(paths[0]) as store:
#     data = store.get(contracts[0])


### ADD FEATURES
# add technical indicators
periods = [5, 30, 60, 150, 300, 480, 2400, 12000]
# data = add_technical_indicators(data, periods=periods)  # delete later
data = tml.modeling.features.add_technical_indicators(data, periods=periods)
data.columns = [cl[0] if isinstance(cl, tuple) else cl for cl in data.columns]

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

# Serial Correlation (Takes time) TO SLOW
# window_autocorr = 50

# data['autocorr_1'] = np.log(data['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=1), raw=False)
# data['autocorr_2'] = np.log(data['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=2), raw=False)
# data['autocorr_3'] = np.log(data['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=3), raw=False)
# data['autocorr_4'] = np.log(data['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=4), raw=False)
# data['autocorr_5'] = np.log(data['close']).diff().rolling(
#     window=window_autocorr, min_periods=window_autocorr,
#     center=False).apply(lambda x: x.autocorr(lag=5), raw=False)

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


### REMOVE NAN FOR INDICATORS
data.isna().sum().sort_values(ascending=False).head(20)
columns_na_below = data.isna().sum() < 12010
data = data.loc[:, columns_na_below]
cols_remove_na = range((np.where(data.columns == 'volume')[0].item() + 1), data.shape[1])
data.dropna(subset=data.columns[cols_remove_na], inplace=True)


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


### SAVE UNSTATIONARY SPY
save_path = 'D:/market_data/usa/ohlcv_features/unstat_' + 'SPY' + '.h5'
with pd.HDFStore(save_path) as store:
    store.put('SPY', data)


### LOAD UNSTAT SPY
# contract = ['SPY']
# with pd.HDFStore(save_path) as store:
#     data = store.get(contract[0])
# data.sort_index(inplace=True)


###  STATIONARITY
ohlc = data[['open', 'high', 'low', 'close']]  # save for later
ohlc.columns = ['open_orig', 'high_orig', 'low_orig', 'close_orig']

# get dmin for every column
stationaryCols, min_d = tml.modeling.stationarity.min_ffd_all_cols(data)

# save to github for later 
min_dmin_d_save_for_backtesting = pd.Series(0, index=data.columns)
min_dmin_d_save_for_backtesting.update(min_d)
min_dmin_d_save_for_backtesting.dropna(inplace=True)
min_dmin_d_save_for_backtesting.to_csv(
    'C:/Users/Mislav/Documents/GitHub/trademl/data/min_d_' + contract + '.csv', sep=';')

# convert unstationary to stationary
data = tml.modeling.stationarity.unstat_cols_to_stat(data, min_d, stationaryCols)  # tml.modeling.stationarity.unstat_cols_to_stat
data.dropna(inplace=True)

# merge orig ohlc to spyStat
data = data.merge(ohlc, how='left', left_index=True, right_index=True)


### REMOVE FEATURES WITH VERY HIGH CORRELATION
# calculate correlation matrix
feature_columns = data.drop(columns=['open', 'high', 'low',
                                     'open_vix', 'high_vix', 'low_vix',
                                     'open_orig', 'high_orig', 'low_orig',
                                     'close_orig']).columns  # remove this columns, not use when calculating corrr
corrs = pd.DataFrame(np.corrcoef(data[feature_columns].values, rowvar=False),
                     columns=data[feature_columns].columns)  # correlation matrix with numpy for performance
corrs.index = corrs.columns  # add row index

# remove sequentally highly correlated features
cols_remove = []
for i, col in enumerate(corrs.columns):
    corrs_sample = corrs.iloc[i:, i:]  # remove ith column and row
    corrs_vec = corrs_sample[col].iloc[(i+1):]
    index_multicorr = corrs_vec.iloc[np.where(np.abs(corrs_vec) >= 0.99)]  # remove features with corr coef > 0.99
    cols_remove.append(index_multicorr)
extreme_correlateed_assets = pd.DataFrame(cols_remove).columns
data = data.drop(columns=extreme_correlateed_assets)



### CLUSTER FEATURES

### CODE  FROM DEVELOP BRANCH TILL IT IS MERGED TO THE MASTER


# import statsmodels.api as sm
# from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import linkage, fcluster
# from statsmodels.regression.linear_model import OLS
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_samples

# from mlfinlab.codependence.information import variation_of_information_score, get_mutual_info
# from mlfinlab.codependence.correlation import distance_correlation
# from mlfinlab.codependence.gnpr_distance import spearmans_rho, gpr_distance, gnpr_distance

# from typing import Union, Iterable, Optional

# # from mlfinlab.codependence.codependence_matrix import get_dependence_matrix, get_distance_matrix


# def get_dependence_matrix(df: pd.DataFrame, dependence_method: str, theta: float = 0.5,
#                           bandwidth: float = 0.01) -> pd.DataFrame:
#     """
#     This function returns a dependence matrix for elements given in the dataframe using the chosen dependence method.
#     List of supported algorithms to use for generating the dependence matrix: ``information_variation``,
#     ``mutual_information``, ``distance_correlation``, ``spearmans_rho``, ``gpr_distance``, ``gnpr_distance``.
#     :param df: (pd.DataFrame) Features.
#     :param dependence_method: (str) Algorithm to be use for generating dependence_matrix.
#     :param theta: (float) Type of information being tested in the GPR and GNPR distances. Falls in range [0, 1].
#                           (0.5 by default)
#     :param bandwidth: (float) Bandwidth to use for splitting observations in the GPR and GNPR distances. (0.01 by default)
#     :return: (pd.DataFrame) Dependence matrix.
#     """
#     # Get the feature names.
#     features_cols = df.columns.values
#     n = df.shape[1]
#     np_df = df.values.T  # Make columnar access, but for np.array

#     # Defining the dependence function.
#     if dependence_method == 'information_variation':
#         dep_function = lambda x, y: variation_of_information_score(x, y, normalize=True)
#     elif dependence_method == 'mutual_information':
#         dep_function = lambda x, y: get_mutual_info(x, y, normalize=True)
#     elif dependence_method == 'distance_correlation':
#         dep_function = distance_correlation
#     elif dependence_method == 'spearmans_rho':
#         dep_function = spearmans_rho
#     elif dependence_method == 'gpr_distance':
#         dep_function = lambda x, y: gpr_distance(x, y, theta=theta)
#     elif dependence_method == 'gnpr_distance':
#         dep_function = lambda x, y: gnpr_distance(x, y, theta=theta, bandwidth=bandwidth)
#     else:
#         raise ValueError(f"{dependence_method} is not a valid method. Please use one of the supported methods \
#                             listed in the docsting.")

#     # Generating the dependence_matrix
#     dependence_matrix = np.array([
#         [
#             dep_function(np_df[i], np_df[j]) if j < i else
#             # Leave diagonal elements as 0.5 to later double them to 1
#             0.5 * dep_function(np_df[i], np_df[j]) if j == i else
#             0  # Make upper triangle 0 to fill it later on
#             for j in range(n)
#         ]
#         for i in range(n)
#     ])

#     # Make matrix symmetrical
#     dependence_matrix = dependence_matrix + dependence_matrix.T

#     #  Dependence_matrix converted into a DataFrame.
#     dependence_df = pd.DataFrame(data=dependence_matrix, index=features_cols, columns=features_cols)

#     if dependence_method == 'information_variation':
#         return 1 - dependence_df  # IV is reverse, 1 - independent, 0 - similar

#     return dependence_df

# def get_distance_matrix(X: pd.DataFrame, distance_metric: str = 'angular') -> pd.DataFrame:
#     """
#     Applies distance operator to a dependence matrix.
#     This allows to turn a correlation matrix into a distance matrix. Distances used are true metrics.
#     List of supported distance metrics to use for generating the distance matrix: ``angular``, ``squared_angular``,
#     and ``absolute_angular``.
#     :param X: (pd.DataFrame) Dataframe to which distance operator to be applied.
#     :param distance_metric: (str) The distance metric to be used for generating the distance matrix.
#     :return: (pd.DataFrame) Distance matrix.
#     """
#     if distance_metric == 'angular':
#         distfun = lambda x: ((1 - x).round(5) / 2.) ** .5
#     elif distance_metric == 'abs_angular':
#         distfun = lambda x: ((1 - abs(x)).round(5) / 2.) ** .5
#     elif distance_metric == 'squared_angular':
#         distfun = lambda x: ((1 - x ** 2).round(5) / 2.) ** .5
#     else:
#         raise ValueError(f'{distance_metric} is a unknown distance metric. Please use one of the supported methods \
#                             listed in the docsting.')

#     return distfun(X).fillna(0)


# def _improve_clusters(corr_mat: pd.DataFrame, clusters: dict, top_clusters: dict) -> Union[
#         pd.DataFrame, dict, pd.Series]:
#     """
#     Improve number clusters using silh scores
#     :param corr_mat: (pd.DataFrame) Correlation matrix
#     :param clusters: (dict) Clusters elements
#     :param top_clusters: (dict) Improved clusters elements
#     :return: (tuple) [ordered correlation matrix, clusters, silh scores]
#     """
#     clusters_new, new_idx = {}, []
#     for i in clusters.keys():
#         clusters_new[len(clusters_new.keys())] = list(clusters[i])

#     for i in top_clusters.keys():
#         clusters_new[len(clusters_new.keys())] = list(top_clusters[i])

#     map(new_idx.extend, clusters_new.values())
#     corr_new = corr_mat.loc[new_idx, new_idx]

#     dist = ((1 - corr_mat.fillna(0)) / 2.0) ** 0.5

#     kmeans_labels = np.zeros(len(dist.columns))
#     for i in clusters_new:
#         idxs = [dist.index.get_loc(k) for k in clusters_new[i]]
#         kmeans_labels[idxs] = i

#     silh_scores_new = pd.Series(silhouette_samples(dist, kmeans_labels), index=dist.index)
#     return corr_new, clusters_new, silh_scores_new


# def _cluster_kmeans_base(corr_mat: pd.DataFrame, max_num_clusters: int = 10, repeat: int = 10) -> Union[
#         pd.DataFrame, dict, pd.Series]:
#     """
#     Initial clustering step using KMeans.
#     :param corr_mat: (pd.DataFrame) Correlation matrix
#     :param max_num_clusters: (int) Maximum number of clusters to search for.
#     :param repeat: (int) Number of clustering algorithm repetitions.
#     :return: (tuple) [ordered correlation matrix, clusters, silh scores]
#     """

#     # Distance matrix

#     # Fill main diagonal of corr matrix with 1s to avoid elements being close to 1 with e-16.
#     # As this previously caused Errors when taking square root from negative values.
#     corr_mat[corr_mat > 1] = 1
#     distance = ((1 - corr_mat.fillna(0)) / 2.0) ** 0.5
#     silh = pd.Series(dtype='float64')

#     # Get optimal num clusters
#     for _ in range(repeat):
#         for num_clusters in range(2, max_num_clusters + 1):
#             kmeans_ = KMeans(n_clusters=num_clusters, n_init=1)
#             kmeans_ = kmeans_.fit(distance)
#             silh_ = silhouette_samples(distance, kmeans_.labels_)
#             stat = (silh_.mean() / silh_.std(), silh.mean() / silh.std())

#             if np.isnan(stat[1]) or stat[0] > stat[1]:
#                 silh = silh_
#                 kmeans = kmeans_

#     # Number of clusters equals to length(kmeans labels)
#     new_idx = np.argsort(kmeans.labels_)

#     # Reorder rows
#     corr1 = corr_mat.iloc[new_idx]
#     # Reorder columns
#     corr1 = corr1.iloc[:, new_idx]

#     # Cluster members
#     clusters = {i: corr_mat.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in
#                 np.unique(kmeans.labels_)}
#     silh = pd.Series(silh, index=distance.index)

#     return corr1, clusters, silh


# def _check_improve_clusters(new_tstat_mean: float, mean_redo_tstat: float, old_cluster: tuple,
#                             new_cluster: tuple) -> tuple:
#     """
#     Checks cluster improvement condition based on t-statistic.
#     :param new_tstat_mean: (float) T-statistics
#     :param mean_redo_tstat: (float) Average t-statistcs for cluster improvement
#     :param old_cluster: (tuple) Old cluster correlation matrix, optimized clusters, silh scores
#     :param new_cluster: (tuple) New cluster correlation matrix, optimized clusters, silh scores
#     :return: (tuple) Cluster
#     """

#     if new_tstat_mean > mean_redo_tstat:
#         return old_cluster
#     return new_cluster


# def cluster_kmeans_top(corr_mat: pd.DataFrame, repeat: int = 10) -> Union[pd.DataFrame, dict, pd.Series, bool]:
#     """
#     Improve the initial clustering by leaving clusters with high scores unchanged and modifying clusters with
#     below average scores.
#     :param corr_mat: (pd.DataFrame) Correlation matrix
#     :param repeat: (int) Number of clustering algorithm repetitions.
#     :return: (tuple) [correlation matrix, optimized clusters, silh scores, boolean to rerun ONC]
#     """
#     # pylint: disable=no-else-return

#     max_num_clusters = min(corr_mat.drop_duplicates().shape[0], corr_mat.drop_duplicates().shape[1]) - 1
#     corr1, clusters, silh = _cluster_kmeans_base(corr_mat, max_num_clusters=max_num_clusters, repeat=repeat)

#     # Get cluster quality scores
#     cluster_quality = {i: float('Inf') if np.std(silh[clusters[i]]) == 0 else np.mean(silh[clusters[i]]) /
#                           np.std(silh[clusters[i]]) for i in clusters.keys()}
#     avg_quality = np.mean(list(cluster_quality.values()))
#     redo_clusters = [i for i in cluster_quality.keys() if cluster_quality[i] < avg_quality]

#     if len(redo_clusters) <= 2:
#         # If 2 or less clusters have a quality rating less than the average then stop.
#         return corr1, clusters, silh
#     else:
#         keys_redo = []
#         for i in redo_clusters:
#             keys_redo.extend(clusters[i])

#         corr_tmp = corr_mat.loc[keys_redo, keys_redo]
#         mean_redo_tstat = np.mean([cluster_quality[i] for i in redo_clusters])
#         _, top_clusters, _ = cluster_kmeans_top(corr_tmp, repeat=repeat)

#         # Make new clusters (improved)
#         corr_new, clusters_new, silh_new = _improve_clusters(corr_mat,
#                                                              {i: clusters[i] for i in clusters.keys() if
#                                                               i not in redo_clusters},
#                                                              top_clusters)
#         new_tstat_mean = np.mean(
#             [np.mean(silh_new[clusters_new[i]]) / np.std(silh_new[clusters_new[i]]) for i in clusters_new])

#         return _check_improve_clusters(new_tstat_mean, mean_redo_tstat, (corr1, clusters, silh),
#                                        (corr_new, clusters_new, silh_new))


# def get_onc_clusters(corr_mat: pd.DataFrame, repeat: int = 10) -> Union[pd.DataFrame, dict, pd.Series]:
#     """
#     Optimal Number of Clusters (ONC) algorithm described in the following paper:
#     `Marcos Lopez de Prado, Michael J. Lewis, Detection of False Investment Strategies Using Unsupervised
#     Learning Methods, 2015 <https://papers.ssrn.com/sol3/abstract_id=3167017>`_;
#     The code is based on the code provided by the authors of the paper.
#     The algorithm searches for the optimal number of clusters using the correlation matrix of elements as an input.
#     The correlation matrix is transformed to a matrix of distances, the K-Means algorithm is applied multiple times
#     with a different number of clusters to use. The results are evaluated on the t-statistics of the silhouette scores.
#     The output of the algorithm is the reordered correlation matrix (clustered elements are placed close to each other),
#     optimal clustering, and silhouette scores.
#     :param corr_mat: (pd.DataFrame) Correlation matrix of features
#     :param repeat: (int) Number of clustering algorithm repetitions
#     :return: (tuple) [correlation matrix, optimized clusters, silh scores]
#     """

#     return cluster_kmeans_top(corr_mat, repeat)


# # pylint: disable=invalid-name
# def get_feature_clusters(X: pd.DataFrame, dependence_metric: str, distance_metric: str = None,
#                          linkage_method: str = None, n_clusters: int = None, critical_threshold: float = 0.0) -> list:
#     """
#     Machine Learning for Asset Managers
#     Snippet 6.5.2.1 , page 85. Step 1: Features Clustering
#     Gets clustered features subsets from the given set of features.
#     :param X: (pd.DataFrame) Dataframe of features.
#     :param dependence_metric: (str) Method to be use for generating dependence_matrix, either 'linear' or
#                               'information_variation' or 'mutual_information' or 'distance_correlation'.
#     :param distance_metric: (str) The distance operator to be used for generating the distance matrix. The methods that
#                             can be applied are: 'angular', 'squared_angular', 'absolute_angular'. Set it to None if the
#                             feature are to be generated as it is by the ONC algorithm.
#     :param linkage_method: (str) Method of linkage to be used for clustering. Methods include: 'single', 'ward',
#                            'complete', 'average', 'weighted', and 'centroid'. Set it to None if the feature are to
#                            be generated as it is by the ONC algorithm.
#     :param n_clusters: (int) Number of clusters to form. Must be less the total number of features. If None then it
#                        returns optimal number of clusters decided by the ONC Algorithm.
#     :param critical_threshold: (float) Threshold for determining low silhouette score in the dataset. It can any real number
#                                 in [-1,+1], default is 0 which means any feature that has a silhouette score below 0 will be
#                                 indentified as having low silhouette and hence requied transformation will be appiled to for
#                                 for correction of the same.
#     :return: (list) Feature subsets.
#     """
#     # Checking if dataset contains features low silhouette
#     X = _check_for_low_silhouette_scores(X, critical_threshold)

#     # Get the dependence matrix
#     if dependence_metric != 'linear':
#         dep_matrix = get_dependence_matrix(X, dependence_method=dependence_metric)
#     else:
#         dep_matrix = X.corr()

#     if n_clusters is None and (distance_metric is None or linkage_method is None):
#         return list(get_onc_clusters(dep_matrix.fillna(0))[1].values())  # Get optimal number of clusters
#     if distance_metric is not None and (linkage_method is not None and n_clusters is None):
#         n_clusters = len(get_onc_clusters(dep_matrix.fillna(0))[1])
#     if n_clusters >= len(X.columns):  # Check if number of clusters exceeds number of features
#         raise ValueError('Number of clusters must be less than the number of features')

#     # Apply distance operator on the dependence matrix
#     dist_matrix = get_distance_matrix(dep_matrix, distance_metric=distance_metric)

#     # Get the linkage
#     link = linkage(squareform(dist_matrix), method=linkage_method)
#     clusters = fcluster(link, t=n_clusters, criterion='maxclust')
#     clustered_subsets = [[f for c, f in zip(clusters, X.columns) if c == ci] for ci in range(1, n_clusters + 1)]

#     return clustered_subsets


# def _cluster_transformation(X: pd.DataFrame, clusters: dict, feats_to_transform: list) -> pd.DataFrame:
#     """
#     Machine Learning for Asset Managers
#     Snippet 6.5.2.1 , page 85. Step 1: Features Clustering (last paragraph)
#     Transforms a dataset to reduce the multicollinearity of the system by replacing the original feature with
#     the residual from regression.
#     :param X: (pd.DataFrame) Dataframe of features.
#     :param clusters: (dict) Clusters generated by ONC algorithm.
#     :param feats_to_transform: (list) Features that have low silhouette score and to be transformed.
#     :return: (pd.DataFrame) Transformed features.
#     """
#     for feat in feats_to_transform:
#         for i, j in clusters.items():

#             if feat in j:  # Selecting the cluster that contains the feature
#                 exog = sm.add_constant(X.drop(j, axis=1)).values
#                 endog = X[feat].values
#                 ols = OLS(endog, exog).fit()

#                 if ols.df_model < (exog.shape[1]-1):
#                     # Degree of freedom is low
#                     new_exog = _combine_features(X, clusters, i)
#                     # Run the regression again on the new exog
#                     ols = OLS(endog, new_exog.reshape(exog.shape[0], -1)).fit()
#                     X[feat] = ols.resid
#                 else:
#                     X[feat] = ols.resid

#     return X


# def _combine_features(X, clusters, exclude_key) -> np.array:
#     """
#     Combines features of each cluster linearly by following a minimum variance weighting scheme.
#     The Minimum Variance weights are calculated without constraints, other than the weights sum to one.
#     :param X: (pd.DataFrame) Dataframe of features.
#     :param clusters: (dict) Clusters generated by ONC algorithm.
#     :param exclude_key: (int) Key of the cluster which is to be excluded.
#     :return: (np.array) Combined features for each cluster.
#     """

#     new_exog = []
#     for i, cluster in clusters.items():

#         if i != exclude_key:
#             subset = X[cluster]
#             cov_matx = subset.cov()  # Covariance matrix of the cluster
#             eye_vec = np.array(cov_matx.shape[1]*[1], float)
#             try:
#                 numerator = np.dot(np.linalg.inv(cov_matx), eye_vec)
#                 denominator = np.dot(eye_vec, numerator)
#                 # Minimum variance weighting
#                 wghts = numerator/denominator
#             except np.linalg.LinAlgError:
#                 # A singular matrix so giving each component equal weight
#                 wghts = np.ones(subset.shape[1]) * (1/subset.shape[1])
#             new_exog.append(((subset*wghts).sum(1)).values)

#     return np.array(new_exog)


# def _check_for_low_silhouette_scores(X: pd.DataFrame, critical_threshold: float = 0.0) -> pd.DataFrame:
#     """
#     Machine Learning for Asset Managers
#     Snippet 6.5.2.1 , page 85. Step 1: Features Clustering (last paragraph)
#     Checks where the dataset contains features low silhouette due one feature being a combination of
#     multiple features across clusters. This is a problem, because ONC cannot assign one feature to multiple
#     clusters and it needs a transformation.
#     :param X: (pd.DataFrame) Dataframe of features.
#     :param critical_threshold: (float) Threshold for determining low silhouette score.
#     :return: (pd.DataFrame) Dataframe of features.
#     """
#     _, clstrs, silh = get_onc_clusters(X.corr())
#     low_silh_feat = silh[silh < critical_threshold].index
#     if len(low_silh_feat) > 0:
#         print(f'{len(low_silh_feat)} feature/s found with low silhouette score {low_silh_feat}. Returning the transformed dataset')

#         # Returning the transformed dataset
#         return _cluster_transformation(X, clstrs, low_silh_feat)

#     print('No feature/s found with low silhouette score. All features belongs to its respective clusters')

#     return X


# feat_subs = get_feature_clusters(data.drop(columns='close_orig'),
#                                  dependence_metric='information_variation',
#                                  distance_metric='angular',
#                                  linkage_method='single',
#                                  n_clusters=4)


# clusters_onc = ml.clustering.get_feature_clusters(
#     data.drop(columns=['open', 'high', 'low',
#                        'open_vix', 'high_vix', 'low_vix',
#                        'open_orig', 'high_orig', 'low_orig',
#                        'close_orig']),
#     dependence_metric='linear',
#     distance_metric=None,
#     linkage_method=None, 
#     n_clusters=None)


### CODE  FROM DEVELOP BRANCH TILL IT IS MERGED TO THE MASTER


### STRUCTURAL BRAKES

# convert data to hourly to make code faster and decrease random component
close_hourly = data['close_orig'].resample('H').last().dropna()
close_hourly = np.log(close_hourly)

# Chow-Type Dickey-Fuller Test
chow = tml.modeling.structural_breaks.get_chow_type_stat(
    series=close_hourly, min_length=10)
breakdate = chow.loc[chow == chow.max()]
data['chow_segment'] = 0
data['chow_segment'].loc[breakdate.index[0]:] = 1
data['chow_segment'] = np.where(data.index < breakdate.index[0], 0, 1)
data['chow_segment'].value_counts()
# close_daily.plot()
# plt.axvline(chow.loc[chow == chow.max()].index, color='red')


### SADF

# from typing import Union, Tuple

# def _lag_df(df: pd.DataFrame, lags: Union[int, list]) -> pd.DataFrame:
#     """
#     Advances in Financial Machine Learning, Snipet 17.3, page 259.
#     Apply Lags to DataFrame
#     :param df: (int or list) Either number of lags to use or array of specified lags
#     :param lags: (int or list) Lag(s) to use
#     :return: (pd.DataFrame) Dataframe with lags
#     """
#     df_lagged = pd.DataFrame()
#     if isinstance(lags, int):
#         lags = range(1, lags + 1)
#     else:
#         lags = [int(lag) for lag in lags]

#     for lag in lags:
#         temp_df = df.shift(lag).copy(deep=True)
#         temp_df.columns = [str(i) + '_' + str(lag) for i in temp_df.columns]
#         df_lagged = df_lagged.join(temp_df, how='outer')
#     return df_lagged

# def _get_y_x(series: pd.Series, model: str, lags: Union[int, list],
#              add_const: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Advances in Financial Machine Learning, Snippet 17.2, page 258-259.
#     Preparing The Datasets
#     :param series: (pd.Series) Series to prepare for test statistics generation (for example log prices)
#     :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
#     :param lags: (int or list) Either number of lags to use or array of specified lags
#     :param add_const: (bool) Flag to add constant
#     :return: (pd.DataFrame, pd.DataFrame) Prepared y and X for SADF generation
#     """
#     series = pd.DataFrame(series)
#     series_diff = series.diff().dropna()
#     x = _lag_df(series_diff, lags).dropna()
#     x['y_lagged'] = series.shift(1).loc[x.index]  # add y_(t-1) column
#     y = series_diff.loc[x.index]

#     if add_const is True:
#         x['const'] = 1

#     if model == 'linear':
#         x['trend'] = np.arange(x.shape[0])  # Add t to the model (0, 1, 2, 3, 4, 5, .... t)
#         beta_column = 'y_lagged'  # Column which is used to estimate test beta statistics
#     elif model == 'quadratic':
#         x['trend'] = np.arange(x.shape[0]) # Add t to the model (0, 1, 2, 3, 4, 5, .... t)
#         x['quad_trend'] = np.arange(x.shape[0]) ** 2 # Add t^2 to the model (0, 1, 4, 9, ....)
#         beta_column = 'y_lagged'  # Column which is used to estimate test beta statistics
#     elif model == 'sm_poly_1':
#         y = series.loc[y.index]
#         x = pd.DataFrame(index=y.index)
#         x['const'] = 1
#         x['trend'] = np.arange(x.shape[0])
#         x['quad_trend'] = np.arange(x.shape[0]) ** 2
#         beta_column = 'quad_trend'
#     elif model == 'sm_poly_2':
#         y = np.log(series.loc[y.index])
#         x = pd.DataFrame(index=y.index)
#         x['const'] = 1
#         x['trend'] = np.arange(x.shape[0])
#         x['quad_trend'] = np.arange(x.shape[0]) ** 2
#         beta_column = 'quad_trend'
#     elif model == 'sm_exp':
#         y = np.log(series.loc[y.index])
#         x = pd.DataFrame(index=y.index)
#         x['const'] = 1
#         x['trend'] = np.arange(x.shape[0])
#         beta_column = 'trend'
#     elif model == 'sm_power':
#         y = np.log(series.loc[y.index])
#         x = pd.DataFrame(index=y.index)
#         x['const'] = 1
#         # TODO: Rewrite logic of this module to avoid division by zero
#         with np.errstate(divide='ignore'):
#             x['log_trend'] = np.log(np.arange(x.shape[0]))
#         beta_column = 'log_trend'
#     else:
#         raise ValueError('Unknown model')

#     # Move y_lagged column to the front for further extraction
#     columns = list(x.columns)
#     columns.insert(0, columns.pop(columns.index(beta_column)))
#     x = x[columns]
#     return x, y


# @njit
# def get_betas(X: np.array, y: np.array) -> Tuple[np.array, np.array]:
#     """
#     Advances in Financial Machine Learning, Snippet 17.4, page 259.
#     Fitting The ADF Specification (get beta estimate and estimate variance)
#     :param X: (pd.DataFrame) Features(factors)
#     :param y: (pd.DataFrame) Outcomes
#     :return: (np.array, np.array) Betas and variances of estimates
#     """

#     # Get regression coefficients estimates
#     xy_ = np.dot(X.T, y)
#     xx_ = np.dot(X.T, X)

#     #   check for singularity
#     det = np.linalg.det(xx_)

#     # get coefficient and std from linear regression
#     if det == 0:
#         b_mean = np.array([[np.nan]])
#         b_var = np.array([[np.nan, np.nan]])
#         return None
#     else:
#         xx_inv = np.linalg.inv(xx_)
#         b_mean = np.dot(xx_inv, xy_)
#         err = y - np.dot(X, b_mean)
#         b_var = np.dot(np.transpose(err), err) / (X.shape[0] - X.shape[1]) * xx_inv  # pylint: disable=E1136  # pylint/issues/3139
#         return b_mean, b_var
    

# @njit
# def _get_sadf_at_t(X: pd.DataFrame, y: pd.DataFrame, min_length: int, model: str, phi: float) -> float:
#     """
#     Advances in Financial Machine Learning, Snippet 17.2, page 258.
#     SADF's Inner Loop (get SADF value at t)
#     :param X: (pd.DataFrame) Lagged values, constants, trend coefficients
#     :param y: (pd.DataFrame) Y values (either y or y.diff())
#     :param min_length: (int) Minimum number of samples needed for estimation
#     :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
#     :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
#     :return: (float) SADF statistics for y.index[-1]
#     """
#     start_points = prange(0, y.shape[0] - min_length + 1)
#     bsadf = -np.inf
#     for start in start_points:
#         y_, X_ = y[start:], X[start:]
#         b_mean_, b_std_ = get_betas(X_, y_)
#         # if b_mean_ is not None:  DOESNT WORK WITH NUMBA
#         b_mean_, b_std_ = b_mean_[0, 0], b_std_[0, 0] ** 0.5
#         # Rewrite logic of this module to avoid division by zero
#         if b_std_ != np.float64(0):
#             all_adf = b_mean_ / b_std_
#         if model[:2] == 'sm':
#             all_adf = np.abs(all_adf) / (y.shape[0]**phi)
#         if all_adf > bsadf:
#             bsadf = all_adf
#     return bsadf


# @njit
# def _sadf_outer_loop(X: np.array, y: np.array, min_length: int, model: str, phi: float,
#                      ) -> pd.Series:
#     """
#     This function gets SADF for t times from molecule
#     :param X: (pd.DataFrame) Features(factors)
#     :param y: (pd.DataFrame) Outcomes
#     :param min_length: (int) Minimum number of observations
#     :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
#     :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
#     :param molecule: (list) Indices to get SADF
#     :return: (pd.Series) SADF statistics
#     """
#     sadf_series_val = []
#     for index in range(1, (y.shape[0]-min_length+1)):
#         X_subset = X[:min_length+index]
#         y_subset = y[:min_length+index]
#         value = _get_sadf_at_t(X_subset, y_subset, min_length, model, phi)
#         sadf_series_val.append(value)
#     return sadf_series_val



# def get_sadf(series: pd.Series, model: str, lags: Union[int, list], min_length: int, add_const: bool = False,
#              phi: float = 0, num_threads: int = 8, verbose: bool = True) -> pd.Series:
    
#     X, y = _get_y_x(series, model, lags, add_const)
#     molecule = y.index[min_length:y.shape[0]]
#     X_val = X.values
#     y_val = y.values
    
#     sadf_series =_sadf_outer_loop(X=X.values, y=y.values,
#                                   min_length=min_length, model=model, phi=phi)
#     sadf_series_val = np.array(sadf_series)
    
#     return sadf_series_val


# # convert data to hourly to make code faster and decrease random component
# close_daily = data['close_orig'].resample('D').last().dropna()
# close_daily = np.log(close_daily)


# series = close_daily.iloc[:2000].copy()
# model = 'linear'
# lags = 2
# min_length = 20
# add_const = False
# phi = 0


# MEASURE PERFORMANCE
# from timeit import default_timer as timer
# from datetime import timedelta

# # MLFINLAB PACKAGE
# start = timer()
# mlfinlab_results = ml.structural_breaks.get_sadf(
#     series, min_length=min_length, model=model, phi=phi, num_threads=1, lags=lags)
# end = timer()
# print(timedelta(seconds=end-start))
# print(mlfinlab_results.shape)
# print(mlfinlab_results.head(20))
# print(mlfinlab_results.tail(20))


# # MY FUNCTION
# start = timer()
# results = get_sadf(
#     close_daily, min_length=20, add_const='True', model='linear', phi=0.5, num_threads=1, lags=2)
# end = timer()
# print(timedelta(seconds=end-start))
# type(results)
# print(results.shape)
# print(results[:20])
# print(results[-25:])


### TREND SCANNING LABELING
# ts_look_forward_window = [60, 60*8, 60*8*5, 60*8*10, 60*8*15, 60*8*20]
# ts_min_sample_length = [5, 60, 60, 60, 60, 60]
# ts_step = [1, 5, 10, 15, 20, 25]

ts_look_forward_window = [60, 60*8]
ts_min_sample_length = [5, 60]
ts_step = [1, 5]

data_sample = data.iloc[:10000]

trernd_scanning_values = []
for tlf, tmins, sstep in zip(ts_look_forward_window, ts_min_sample_length, ts_step):
    trend_scanning_pipe = tml.modeling.pipelines.TrendScanning(
        close_name='close_orig',
        volatility_lookback=tb_volatility_lookback,
        volatility_scaler=tb_volatility_scaler,
        ts_look_forward_window=ts_look_forward_window,
        ts_min_sample_length=ts_min_sample_length,
        ts_step=ts_step
    )
    labeling_info = trend_scanning_pipe.fit(data_sample)
    trernd_scanning_values.append(trend_scanning_pipe.transform(data_sample))

    

labeling_info = trend_scanning_pipe.fit(data)
X = trend_scanning_pipe.transform(data)


### SAVE TO DATABASE
# add to database
# data['date'] = data.index
# tml.modeling.utils.write_to_db(data.iloc[:50000], "odvjet12_ml_data_usa", 'SPY')
# write_to_db_update(data.iloc[50000:100000], "odvjet12_ml_data_usa", 'SPY')
# seq = np.append(np.arange(1561001, data.shape[0], 50000), (data.shape[0]+1))
# for index, i in enumerate(seq):
#     print(seq[index], seq[index+1])
#     write_to_db_update(data.iloc[seq[index]:seq[index+1]], "odvjet12_ml_data_usa", 'SPY')
    

### SAVE SPY WITH VIX

# save SPY
save_path = 'D:/market_data/usa/ohlcv_features/' + 'SPY' + '.h5'
with pd.HDFStore(save_path) as store:
    store.put('SPY', data)


# import pandas as pd
# import numpy as np
# import mlfinlab as ml

# url = 'https://raw.githubusercontent.com/MislavSag/trademl/master/trademl/modeling/random_forest/X_TEST.csv'
# X_TEST = pd.read_csv(url, sep=',')
# feat_subs = get_feature_clusters(
#     X_TEST.iloc[:,1:], dependence_metric='information_variation',
#     distance_metric='angular', linkage_method='single', n_clusters=4)

# clusters_onc = get_feature_clusters(
#     X_TEST.iloc[:,1:],
#     dependence_metric='mutual_information',
#     distance_metric=None,
#     linkage_method=None, 
#     n_clusters=None)

# hierarchical_clusters = get_feature_clusters(
#     X_TEST.iloc[:,1:],
#     dependence_metric='mutual_information',
#     distance_metric=None,
#     linkage_method=None, 
#     n_clusters=None)

# hierarchical_clusters = get_feature_clusters(
#     X_TEST.iloc[:,1:],
#     dependence_metric='linear',
#     distance_metric='angular',
#     linkage_method='single',
#     n_clusters=None)
