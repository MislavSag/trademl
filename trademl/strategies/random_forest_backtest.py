import backtrader as bt
import argparse
from pathlib import Path
import pandas as pd
import datetime
import joblib
import mlfinlab as ml
import trademl as tml


class RandomForestStrategy(bt.Strategy):
    
    params = (
        ('volatility_scaler', 1),
        ('volatility_lookback', 50)
    )
    
    def start(self):
        # get started value
        self.val_start = self.broker.get_cash()  # keep the starting cash
        self.log(f"Type:  {type(self.datas[0].close)}")

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()}, {txt}')
        print(f'{self.datas[0].datetime.datetime(0)}')

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        
        # load ml model
        clf = joblib.load("C:/Users/Mislav/Documents/GitHub/trademl/trademl/modeling/rf_model.pkl")
        
        # Compute volatility get CUSUM events
        self.daily_vol = ml.util.get_daily_vol(
            pd.Series(self.datas[0].close, index=pd.Timestamp(self.datas[0].datetime.datetime(0))),
            lookback=self.params.volatility_lookback)
        # self.cusum_events = ml.filters.cusum_filter(
        #     self.dataclose,
        #     threshold=self.daily_vol.mean()*self.params.volatility_scaler)
        
        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
    def notify_order(self, order):
        
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED VIX Price: {order.executed.price}\
                    Cost: {order.executed.value} Comm:{order.executed.comm}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(f'SELL EXECUTED VIX Price: {order.executed.price}\
                    Cost: {order.executed.value} Comm:{order.executed.comm}')
                # self.bar_executed = len(self)
               
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
        # self.log(f'Volatility {self.daily_vol[0]}')

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            # self.log(f'To see what is in orer, when buy/close {self.order}')
            return
        
        # Check if we are in the market
        if not self.position:
            # Not yet ... we MIGHT BUY if ...
            if self.data.close[0] > (1.05 * self.data.open[0]):  # and len(self) > (self.bar_executed + 5)
            # if self.cusum_events[0] == self.datas[0].datetime.datetime(0):  # and len(self) > (self.bar_executed + 5)
                self.log(f"Time of CUSUM event: {self.data.close[0]}")
                # self.log(f"Time of CUSUM event: {self.cusum_events[0]}")
                # self.log(f"BUY SPY AT VIX VALUE: {self.vixclose[0]}")
                # self.order = self.buy()
                
        # elif self.vixclose[0] > self.params.exitvix:
        #     self.log(f"SELL SPY AT VIX VALUE: {self.vixclose[0]}")
        #     self.order = self.sell()
            
    def stop(self):
        # calculate the actual returns
        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        print('ROI:        {:.2f}%'.format(100.0 * self.roi))


# Pandas data feed
path_ = "C:/Users/Mislav/Documents/GitHub/trademl/trademl/modeling/random_forest/feature_names.csv"
feature_names = pd.read_csv(path_)
feature_names = [(value, value) for index, value in feature_names.squeeze().items()]
class PandasData(bt.feeds.PandasData):
    '''
    The ``dataname`` parameter inherited from ``feed.DataBase`` is the pandas
    DataFrame
    '''
    path_ = "C:/Users/Mislav/Documents/GitHub/trademl/trademl/modeling/random_forest/feature_names.csv"
    feature_names = pd.read_csv(path_)
    # params = (
    #     # Possible values for datetime (must always be present)
    #     #  None : datetime is the "index" in the Pandas Dataframe
    #     #  -1 : autodetect position or case-wise equal name
    #     #  >= 0 : numeric index to the colum in the pandas dataframe
    #     #  string : column name (as index) in the pandas dataframe
    #     ('datetime', None),
        
    #     # Possible values below:
    #     #  None : column not present
    #     #  -1 : autodetect position or case-wise equal name
    #     #  >= 0 : numeric index to the colum in the pandas dataframe
    #     #  string : column name (as index) in the pandas dataframe
    #     ('open', -1),
    #     ('high', -1),
    #     ('low', -1),
    #     ('close', -1),
    #     ('volume', -1),
    #     ('openinterest', -1), 
    #     ('vixClose', -1)  
    # )
    params = feature_names



cerebro = bt.Cerebro()
DATA_PATH = 'D:/market_data/usa/ohlcv_features/'
contract = 'SPY'
# with pd.HDFStore(DATA_PATH + contract + '.h5') as store:
#     df = store.get('spy_with_vix')  # 'spy_with_vix'
# df.sort_index(inplace=True)
# df['openinterest'] = 0
# df.index = df.index.rename('datetime')
df_test = df.iloc[:50]
data = PandasData(dataname=df_test)
cerebro.adddata(data)
stclass = RandomForestStrategy
cerebro.addstrategy(stclass)
cerebro.addsizer(bt.sizers.FixedSize)
cerebro.run()


# def run(args=None):
    
#     args = parse_args(args)

#     cerebro = bt.Cerebro()

#     # Data feed kwargs
#     kwargs = dict(**eval('dict(' + args.dargs + ')'))

#     # Parse from/to-date
#     dtfmt, tmfmt = '%Y-%m-%d', 'T%H:%M:%S'
#     for a, d in ((getattr(args, x), x) for x in ['fromdate', 'todate']):
#         if a:
#             strpfmt = dtfmt + tmfmt * ('T' in a)
#             kwargs[d] = datetime.datetime.strptime(a, strpfmt)

#     # import data
#     data_store = Path(args.data)
#     DATA_PATH = 'D:/market_data/usa/ohlcv_features/'
#     contract = 'SPY'
#     with pd.HDFStore(DATA_PATH + contract + '.h5') as store:
#         df = store.get('spy_with_vix')  # 'spy_with_vix'
#     df.sort_index(inplace=True)
               
#     df = df.iloc[:int(args.nos)]
#     df['openinterest'] = 0
#     df.index = df.index.rename('datetime')
#     df = df.iloc[:10000]
    
#     data = PandasData(dataname=df)
#     cerebro.adddata(data)

#     # Strategy
#     if args.vix_sma:
#         stclass = RandomForestStrategy
#     # elif args.vix_fix:
#     #     stclass = FixedVixStrategy

#     cerebro.addstrategy(stclass, **eval('dict(' + args.strat + ')'))

#     # Broker
#     broker_kwargs = dict(coc=True)  # default is cheat-on-close active
#     broker_kwargs.update(eval('dict(' + args.broker + ')'))
#     cerebro.broker = bt.brokers.BackBroker(**broker_kwargs)

#     # Sizer
#     cerebro.addsizer(bt.sizers.FixedSize, **eval('dict(' + args.sizer + ')'))

#     # Execute
#     cerebro.run(**eval('dict(' + args.cerebro + ')'))

#     if args.plot:  # Plot if requested to
#         cerebro.plot(**eval('dict(' + args.plot + ')'))
    

# def parse_args(pargs=None):
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#         description=(
#             'Backtrader VIX script'
#         )
#     )
    
#     parser.add_argument('--data', default='../../datas/2005-2006-day-001.txt',
#                         required=False, help='Data to read in')

#     parser.add_argument('--dargs', required=False, default='',
#                         metavar='kwargs', help='kwargs in key=value format')

#     # Defaults for dates
#     parser.add_argument('--fromdate', required=False, default='',
#                         help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')

#     parser.add_argument('--todate', required=False, default='',
#                         help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')

#     parser.add_argument('--nos', required=False, default=10000,
#                         metavar='kwargs', help='kwargs in key=value format')

#     parser.add_argument('--cerebro', required=False, default='',
#                         metavar='kwargs', help='kwargs in key=value format')

#     parser.add_argument('--broker', required=False, default='',
#                         metavar='kwargs', help='kwargs in key=value format')

#     parser.add_argument('--sizer', required=False, default='',
#                         metavar='kwargs', help='kwargs in key=value format')

#     parser.add_argument('--strat', '--strategy', required=False, default='',
#                         metavar='kwargs', help='kwargs in key=value format')

#     parser.add_argument('--plot', required=False, default='',
#                         nargs='?', const='{}',
#                         metavar='kwargs', help='kwargs in key=value format')

#     pgroup = parser.add_mutually_exclusive_group(required=True)
#     pgroup.add_argument('--vix-sma', required=False, action='store_true',
#                         help='VIX crosoover')

#     pgroup.add_argument('--vix-fix', required=False, action='store_true',
#                         help='Sell if VIX grater than vixmax')

#     return parser.parse_args(pargs)


# if __name__ == '__main__':
#     run()
