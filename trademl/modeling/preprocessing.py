import pandas as pd
import numpy as np



def remove_correlated_columns(data, columns_ignore, threshold=0.99):
    """
    Remove correlated features from the pandas dataframe.
    """
    # calculate correlation matrix
    corrs = pd.DataFrame(np.corrcoef(
        data.drop(columns=columns_ignore).values, rowvar=False),
                         columns=data.drop(columns=columns_ignore).columns)
    corrs.index = corrs.columns  # add row index
    # remove sequentally highly correlated features
    cols_remove = []
    for i, col in enumerate(corrs.columns):
        corrs_sample = corrs.iloc[i:, i:]  # remove ith column and row
        corrs_vec = corrs_sample[col].iloc[(i+1):]
        index_multicorr = corrs_vec.iloc[np.where(np.abs(corrs_vec) >= threshold)]
        cols_remove.append(index_multicorr)
    extreme_correlateed_assets = pd.DataFrame(cols_remove).columns
    data = data.drop(columns=extreme_correlateed_assets)
    
    return data
