import os
from pathlib import Path
import pandas as pd
import trademl as tml
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from trademl.modeling.data_import import import_ohlcv
from trademl.modeling.outliers import RemoveOutlierDiffMedian
from trademl.modeling.features import AddFeatures
from trademl.modeling.stationarity import Fracdiff


# Parameters
save_path = 'D:/market_data/usa/ohlcv_features'
contract = 'SPY_IB'
keep_unstationary = True

# Import data
data = import_ohlcv(save_path, contract=contract)

# Preprocessing
pipe = make_pipeline(
    RemoveOutlierDiffMedian(median_outlier_thrteshold=25),
    AddFeatures(ta_periods=[10, 20]),
    Fracdiff(keep_unstationary=keep_unstationary)
    )
X = pipe.fit_transform(data)

# Save localy
save_path_local = os.path.join(Path(save_path), contract + '_clean' + '.h5')
if os.path.exists(save_path_local):
    os.remove(save_path_local)
with pd.HDFStore(save_path_local) as store:
    store.put(contract + '_clean', X)
# save to mfiles
# if env_directory is not None:
#     mfiles_client = tml.modeling.utils.set_mfiles_client(env_directory)
#     tml.modeling.utils.destroy_mfiles_object(mfiles_client, [file_name + '.h5'])
#     wd = os.getcwd()
#     os.chdir(Path(save_path))
#     mfiles_client.upload_file(file_name + '.h5', object_type='Dokument')
#     os.chdir(wd)
