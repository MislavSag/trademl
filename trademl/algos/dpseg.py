import datetime
import os.path
from pathlib import Path
import sys
import backtrader as bt
import pandas as pd
import requests
import json
import alpaca_backtrader_api
import argparse
from mlfinlab.structural_breaks import get_sadf
# import quantstats as qs
# import webbrowser


# MOVE LATER TO ENV FOLDER
ALPACA_API_KEY = 'PKEG3YILHHTMTC2I6B08'
ALPACA_SECRET_KEY = 'e3yYRhb0EHj8FpLlAFNRFShy6i5b2oaSZLiDuRKF'
USE_POLYGON = False


# Radf indicator using R
class Dpseg(bt.Indicator):
    lines = ('dpseg',)
    params = dict(period=100)
    
    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        self.close_slice = self.data0.close.get(size=self.p.period)
        self.time_slice = self.data0.datetime.get(size=self.p.period)
        # self.time_slice = [t_.timestamp() for t_ in self.time_slice]
        x = {
            'time': self.time_slice.tolist(),
            'price': self.close_slice.tolist(),
            'type_': 'var'
            # 'p': 0.5
        }
        x = json.dumps(x)
        res = requests.post("http://46.101.219.193/alphar/dpseg", data=x)
        res_json = res.json()
        # print(f'last slope {res_json}')
        self.lines.dpseg[0] = res_json[0]
    
    plotinfo = dict(plot=False, subplot=False)


class CashMarket(bt.analyzers.Analyzer):
  def start(self):
    super(CashMarket, self).start()

  def create_analysis(self):
    self.rets = {}
    self.vals = 0.0

  def notify_cashvalue(self, cash, value):
    self.vals = (cash, value)
    self.rets[self.strategy.datetime.datetime().strftime("%Y-%m-%d")] = self.vals

  def get_analysis(self):
    return self.rets


# Create a Stratey
class TestStrategy(bt.Strategy):
    params = dict(
        maperiod=15,
        printlog=False,
        exectype=bt.Order.Market,
        # stake=10,
        # stopafter=0,
        # valid=None,
        # cancel=0,
        # donotcounter=False,
        # sell=False,
        # usebracket=False,
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.datetime(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        # self.sma = bt.indicators.SimpleMovingAverage(
        #     self.datas[0], period=self.params.maperiod, plot=False, subplot=False)
        self.dpseg = Dpseg(self.data)

    def start(self):
        self.cash_start = self.broker.get_cash()

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

        # used from here:
        # https://github.com/alpacahq/alpaca-backtrader-api/blob/master/sample/strategy_multiple_indicators.py
        self.log("placing trade for {}. target size: {}".format(
            trade.getdataname(),
            trade.size))

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.dataclose[0], doprint=True)

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            # if self.dataclose[0] > self.sma[0]:
            if self.dpseg[0] > 0:
                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0], doprint=True)

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy(exectype=self.p.exectype)

        else:

            # if self.dataclose[0] < self.sma[0]:
            if self.dpseg[0] < 0:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0], doprint=True)

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell(exectype=self.p.exectype)
                

    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f' %
                 (self.params.maperiod, self.broker.getvalue()), doprint=True)
        # calculate the actual returns
        self.roi = (self.broker.get_value() / self.cash_start) - 1.0
        # self.roi_hold = (self.data0.close[0] / self.start_close) - 1.0
        print('ROI:        {:.2f}%'.format(100.0 * self.roi))
        # print('First close:        {:.2f}'.format(self.start_close))


TIMEFRAMES = {
    None: None,
    'minutes': bt.TimeFrame.Minutes,
    'days': bt.TimeFrame.Days,
    'weeks': bt.TimeFrame.Weeks,
    'months': bt.TimeFrame.Months,
    'years': bt.TimeFrame.Years,
    'notimeframe': bt.TimeFrame.NoTimeFrame,
}


def run(args=None):

    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # parse args
    args = parse_args(args)

    # Definetrade option    
    """
    You have 3 options: 
    - backtest (IS_BACKTEST=True, IS_LIVE=False)
    - paper trade (IS_BACKTEST=False, IS_LIVE=False) 
    - live trade (IS_BACKTEST=False, IS_LIVE=True) 
    """
    broker = args.broker
    IS_BACKTEST = args.isbacktest
    IS_LIVE = args.islive
    if broker == 'alpaca':
        symbol = "T"
    elif broker == 'ib':
        symbol = "AMZN-STK-SMART-USD"

    # Add a strategy and analyzer
    cerebro.addstrategy(TestStrategy)
    # cerebro.addanalyzer(CashMarket, _name='cashmarket')
    
    # Data API store
    if broker == 'alpaca':
        store = alpaca_backtrader_api.AlpacaStore(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=IS_LIVE,
            usePolygon=USE_POLYGON
        )
        DataFactory = store.getdata  # or use alpaca_backtrader_api.AlpacaData
    elif broker == 'ib':
        # IB store
        store=bt.stores.IBStore(host="127.0.0.1", port=7496, clientId=1)

    # Data feed args
    from_date = args.fromdate
    from_date = datetime.datetime.strptime(from_date, '%Y-%m-%d')
    timeframe = bt.TimeFrame.TFrame(args.timeframedata)
    compression = args.compression
   
    # Data feed
    if IS_BACKTEST:
        stockkwargs = dict(
            timeframe=bt.TimeFrame.Minutes,
            historical=True,
            fromdate=from_date,
            # todate=datetime.datetime(2020, 10, 10),
            compression=1
        )
        if broker == 'alpaca':
            data0 = DataFactory(dataname=symbol, **stockkwargs)
        elif broker == 'ib':
            data0 = store.getdata(dataname=symbol, **stockkwargs)
        cerebro.resampledata(data0, timeframe=bt.TimeFrame.Minutes, compression=60*4)
    else:
        stockkwargs = dict(
            timeframe=bt.TimeFrame.Seconds,
            historical=False,  # only historical download
            qcheck=2.0,  # timeout in seconds (float) to check for events
            # fromdate=from_date,  # get data from..
            # todate=datetime.datetime(2020, 9, 22),  # get data from..
            # latethrough=False,  # let late samples through
            # tradename=None  # use a different asset as order target
        )
        data0 = store.getdata(dataname=symbol, **stockkwargs)
        cerebro.resampledata(data0, timeframe=timeframe, compression=compression)
        # or just alpaca_backtrader_api.AlpacaBroker()
        if IS_LIVE:
            broker = store.getbroker()
            cerebro.setbroker(broker)

    # # Add dat to cerebro
    # cerebro.adddata(data0)
    
    # set cash if backtest
    if IS_BACKTEST:
        # backtrader broker set initial simulated cash
        cerebro.broker.setcash(100000.0)       

    # Set sizer
    cerebro.addsizer(bt.sizers.AllInSizer)
    # cerebro.broker.set_checksubmit(checksubmit=False)

    # # check returns and banchmarks
    # if args.timereturn:
    #     cerebro.addobserver(bt.observers.TimeReturn,
    #                         timeframe=TIMEFRAMES[args.timeframe])
    # else:
    #     benchdata = data0
    #     if args.benchdata1:
    #         data1 = DataFactory(dataname=symbol, historical=True,
    #                             fromdate=from_date,
    #                             timeframe=bt.TimeFrame.Minutes, compression=60)
    #         cerebro.adddata(data1, name='Data1')
    #         benchdata = data1

    #     cerebro.addobserver(bt.observers.Benchmark,
    #                         data=benchdata,
    #                         timeframe=TIMEFRAMES[args.timeframe])

    # Run over everything
    cerebro.run(maxcpus=2)

    # # ---- Format the values from results ----
    # df_values = pd.DataFrame(results[0].analyzers.getbyname("cashmarket").get_analysis()).T
    # df_values = df_values.iloc[:, 1]
    # returns = qs.utils.to_returns(df_values)
    # returns.index = pd.to_datetime(returns.index)
    # # ----------------------------------------
    
    # qs.extend_pandas()
    # qs.reports.html(returns, output="qs.html")
    # webbrowser.open('qs.html')

    # Plot
    cerebro.plot()


def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            'Backtrader exuber'
        )
    )
        
    # Defaults for dates
    parser.add_argument('--broker', 
                        default='ib',
                        choices=['ib', 'alpaca'],
                        required=True,
                        action='store',
                        help='Broker to use: alpaca or IB'                        
                        )
    
    parser.add_argument('--timeframedata', default='Minutes',
                        choices=bt.TimeFrame.Names,
                        required=False, action='store',
                        help='TimeFrame for Resample/Replay')
    
    parser.add_argument('--compression', default=1, type=int,
                        required=False, action='store',
                        help='Compression for Resample/Replay')
    
    parser.add_argument('--isbacktest',
                        required=False, action='store_true',
                        help='Do only historical download')
    
    parser.add_argument('--islive',
                        required=False, action='store_true',
                        help='Do only historical download')

    # parser.add_argument('--historical',
    #                     required=False, action='store_true',
    #                     help='Do only historical download')
        
    parser.add_argument('--fromdate', required=False, default='2020-05-01',
                        help='Date[time] in YYYY-MM-DD[THH:MM:SS] format')
    
    parser.add_argument('--benchdata1', required=False, action='store_true',
                        help=('Benchmark against data1'))
        
    parser.add_argument('--timereturn', required=False, action='store_true',
                        default=None,
                        help=('Use TimeReturn observer instead of Benchmark'))

    parser.add_argument('--timeframe', required=False, action='store',
                        default=None, choices=TIMEFRAMES.keys(),
                        help=('TimeFrame to apply to the Observer'))
    
    return parser.parse_args(pargs)


if __name__ == '__main__':
    run()

#  Backtest
# python .\dpseg.py --broker alpaca --isbacktest --timeframedata Minutes --compression 60 --fromdate '2020-06-01'
# Paper
# python .\dpseg_alpaca.py --timeframealpaca Minutes --compression 15 --timeframe years
# Live
# python .\dpseg_alpaca.py --islive --timeframealpaca Minutes --compression 15 --timeframe years
