from sktime.utils.data_container import detabularize



input_data_path = 'D:/market_data/usa/ohlcv_features'
output_data_path = 'D:/algo_trading_files'

### IMPORT DATA
def import_data(data_path, remove_cols, contract='SPY'):
    # import data
    with pd.HDFStore(os.path.join(data_path, contract + '.h5')) as store:
        data = store.get(contract)
    data.sort_index(inplace=True)
    
    # remove variables
    remove_cols = [col for col in remove_cols if col in data.columns]
    data.drop(columns=remove_cols, inplace=True)
    
    return data


data = import_data(input_data_path, [], contract='SPY_raw')

price = pd.DataFrame(data['orig_close'])
X = detabularize(price)
X.iloc[0]

from sktime.datasets import load_arrow_head
X, y = load_arrow_head(return_X_y=True)
X.iloc[0]
type(X)