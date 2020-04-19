import pandas as pd
import h2o


def cbind_pandas_h2o(X_train, y_train):
    """
    Convert padnas df to h2o df and cbind X and y.

    :param X_train: (pd.DataFrame) pandas data frame X
    :param y_train: (pd.Series) pandas Sereis <_train
    :return: (h2o.frame.H2Oframe) merged X and y h2o df
    """
    X_train_h2o = h2o.H2OFrame(X_train)
    y_train_h2o = h2o.H2OFrame(y_train.to_frame())
    train = X_train_h2o.cbind(y_train_h2o)
    return train
