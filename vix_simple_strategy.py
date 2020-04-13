import backtrader as bt
from backtrader_data import PandasData
import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
from pathlib import Path
import argparse
import pandas as pd
import tables
from test_strategies import (
    TestStrategy,
    TestStrategyIndicators,
    SmaOptimizationStrategy)
from vix_strategies import SmaVixStrategy, FixedVixStrategy


if __name__ == '__main__':
    
    # import data
    data_store = Path('C:/Users/Mislav/algoAItrader/data/spy.h5')
    with pd.HDFStore(data_store) as store:
        df = store.get('spy')
    df = df.iloc[:10000]
    df['openinterest'] = 0
    df.index = df.index.rename('datetime')
    print(df.head())
    df['vixClose'].describe()

    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    # strats = cerebro.optstrategy(
    #     SmaOptimizationStrategy,
    #     maperiod=range(10, 12))
    cerebro.addstrategy(FixedVixStrategy, exitvix = 30)

    # # Datas are in a subfolder of the samples. Need to find where the script is
    # # because it could have been called from anywhere
    # modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    # datapath = os.path.join(modpath, '../../datas/orcl-1995-2014.txt')

    # Create a Data Feed
    data = PandasData(dataname=df)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    # Run over everything
    cerebro.run()
    
    cerebro.plot()
    