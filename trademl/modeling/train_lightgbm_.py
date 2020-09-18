from pathlib import Path
from tensorboardX import SummaryWriter
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import mlfinlab as ml
import trademl as tml
import mfiles
from datetime import datetime
matplotlib.use("Agg")  # don't show graphs because thaty would stop guildai script


### TENSORBORADX WRITER
log_dir = os.getenv("LOGDIR") or "logs/projector/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)


### HYPERPARAMETERS
# load and save data
input_data_path = 'D:/market_data/usa/ohlcv_features'
output_data_path = 'D:/algo_trading_files'
env_directory = None # os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
# features
include_ta = True
# stationarity
stationarity_tecnique = 'fracdiff'
# structural breaks
structural_break_regime = 'all'
# labeling
label_tuning = False
label = 'day_30'  # 'day_1' 'day_2' 'day_5' 'day_10' 'day_20' 'day_30' 'day_60'
labeling_technique = 'trend_scanning'
tb_triplebar_num_days = 10
tb_triplebar_pt_sl = [1, 1]
tb_triplebar_min_ret = 0.004
ts_look_forward_window = 1200  # 60 * 8 * 10 (10 days)
ts_min_sample_length = 30
ts_step = 5
tb_min_pct = 0.10
# filtering
tb_volatility_lookback = 50
tb_volatility_scaler = 1
# feature engineering
correlation_threshold = 0.99
pca = True
# scaling
scaling = 'none'
# performance
num_threads = 1

