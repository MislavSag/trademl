import datetime
import os.path
from pathlib import Path
import sys
import backtrader as bt
import pandas as pd
import requests
import json


# Define indicator
class Radf(bt.Indicator):
    lines = ('radf',)
    params = dict(period=600, adf_lag=2)
    
    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        self.close_slice = self.data0.close.get(size=self.p.period)
        x = {
            'x': self.close_slice.tolist(),
            'adf_lag': self.p.adf_lag
        }
        x = json.dumps(x)
        res = requests.post("http://46.101.219.193/plumber_test/radf", data=x)
        res_json = res.json()
        bsadf = res_json['bsadf']
        bsadf = pd.DataFrame.from_dict(bsadf)
        bsadf_last = bsadf.iloc[-1]
        # print(f'{bsadf_last}')
        self.lines.radf[0] = bsadf_last.item()
        


# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
        ('printlog', False),
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)
        self.radf = Radf(self.data)


    def nextstart(self):
        size = int(self.broker.get_cash() / self.data)
        self.buy(size=size)
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            # if self.dataclose[0] > self.sma[0]:
            if self.radf[0] < 1.012245:
                print(f'Buy at {self.radf[0]}')

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            # if self.dataclose[0] < self.sma[0]:
            if self.radf[0] > 1.012245:
                print(f'Sell at {self.radf[0]}')
                
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f' %
                 (self.params.maperiod, self.broker.getvalue()), doprint=True)


class PandasData(bt.feeds.PandasData):
    '''
    The ``dataname`` parameter inherited from ``feed.DataBase`` is the pandas
    DataFrame
    '''

    params = (
        # Possible values for datetime (must always be present)
        #  None : datetime is the "index" in the Pandas Dataframe
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('datetime', None),

        # Possible values below:
        #  None : column not present
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        # ('volume', -1),
        # ('average', -1),
        # ('barCount', -1),
    )
    
    
if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    save_path = 'D:/market_data/usa/ohlcv_features'  # os.path.abspath(sys.argv[0])
    contract = 'SPY_IB'
    data_path = os.path.join(Path(save_path), 'cache', contract + '.h5')
    
    # Import data
    data = pd.read_hdf(data_path, contract)
    data = data[['open', 'high', 'low', 'close']]
    # change frequency
    data = data.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})  # wow!
    data = data.dropna()
    # data = data.iloc[:2000]

    print('--------------------------------------------------')
    print(data)
    print('--------------------------------------------------')
    
    data = bt.feeds.PandasData(dataname=data)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000.0)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    # Run over everything
    cerebro.run(maxcpus=8)
    

# import rpy2
# print(rpy2.__version__)

# from rpy2.robjects.packages import importr

