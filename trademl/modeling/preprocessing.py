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
        corrs_vec = corrs_sample.iloc[0, 1:]
        index_multicorr = corrs_vec.iloc[np.where(np.abs(corrs_vec) >= threshold)]
        cols_remove.append(index_multicorr)
    extreme_correlateed_assets = pd.DataFrame(cols_remove).columns
    data = data.drop(columns=extreme_correlateed_assets)
    
    return data


def sequence_from_array(data, target_vec, cusum_events, time_step_length):
    """
    Return 3d sequence from matrix that contain features and targets,
    where trading dats are filteres.
    """
    cusum_events_ = cusum_events.intersection(data.index)
    lstm_sequences = []
    targets = []
    for date in cusum_events_:
        observation = data[:date].iloc[-time_step_length:]
        if observation.shape[0] < time_step_length or data.index[-1] < date:
            next
        else:
            lstm_sequences.append(observation.values.reshape((1, observation.shape[0], observation.shape[1])))
            targets.append(target_vec[target_vec.index == date])
    lstm_sequences_all = np.vstack(lstm_sequences)
    targets = np.vstack(targets)
    targets = targets.astype(np.int64)
    return lstm_sequences_all, targets


def scale_expanding(X_train, y_train, X_test, y_test, expand_function):
    """[summary]

    Args:
        X_train (pd.DataFrame): [description]
        y_train (pd.Series): [description]
        X_test (pd.DataFrame): [description]
        y_test (pd.Series): [description]
        expand_function (function): [description]

    Returns:
        pd.Dataframe: [description]
    """
    X_train = X_train.apply(expand_function)
    X_test = X_test.apply(expand_function)
    y_train = y_train.loc[~X_train.isna().any(axis=1)]
    X_train = X_train.dropna()
    y_test = y_test.loc[~X_test.isna().any(axis=1)]
    X_test = X_test.dropna()

    return X_train, X_test, y_train, y_test
