library(modeltime)
library(timetk)
library(reticulate)
library(anytime)
library(RMySQL)
library(DBI)
library(lubridate)
library(rsample)
library(parsnip)
library(dplyr)
library(RemixAutoML)
library(data.table)
library(reticulate)
pd <- reticulate::import('pandas')


# GLOBALS -----

DATA_PATH = 'D:/market_data/usa/ohlcv_features/SPY_raw.h5'
PYTHON_PATH = 'C:/ProgramData/Anaconda3/python.exe'



# IMPORT PYTHON PACKAGES -----

use_python(PYTHON_PATH, TRUE)
pd <- reticulate::import('pandas')
tml <- reticulate::import('trademl')


# UTILS -----

# connect to database
connection <- function() {
  con <- DBI::dbConnect(RMySQL::MySQL(),
                        host = "91.234.46.219",
                        port = 3306L,
                        dbname = "odvjet12_market_data_usa",
                        username = "odvjet12_mislav",
                        password = "Theanswer0207",
                        Trusted_Connection = "True")
}



# IMPORT DATA -----

# import hdf5 data using pandas function (I tried with rhdf5 function too, but without success)
# stock <- pd$read_hdf(DATA_PATH, 'SPY_raw')  # time consuming and memory intensive
# close <- data.frame(datetime = rownames(stock), close = stock$close)
# close$datetime <- anytime::anytime(close$datetime)

# import data from database
db <- connection()
stock <- dbGetQuery(db, 'SELECT date, Open AS Open, high As High, low AS Low, close AS Close FROM SPY')
invisible(dbDisconnect(db))
stock$date <- anytime::anytime(stock$date)
stock_xts <- xts::xts(stock[, -1], order.by=stock$date, unique=TRUE, tzone='America/Chicago')

# import bigdataset
stock <- pd$read_hdf('D:/market_data/usa/ohlcv_features/SPY.h5', 'SPY')
stock_sample <- stock[1:1000, 1:100]
stock_sample <- as.matrix(stock_sample)
mod1<-constructModel(stock_sample,p=4,"Basic",gran=c(150,10),RVAR=FALSE,h=1,cv="Rolling",MN=FALSE,verbose=FALSE,IC=TRUE)
results=cv.BigVAR(mod1)
results
str(results)
plot(results)

SparsityPlot.BigVAR.results(results)
predict(results,n.ahead=1)

# PREPARE DATA -----


# funnction for removing outliers
remove_ourlier_diff_median <- function(data, median_scaler=25) {
  
  # calcualte daily and data differences
  data_daily <- xts::to.daily(data)
  daily_diff <- (na.omit(diff(data_daily)) + 0.005) * median_scaler
  data_diff <- na.omit(diff(data))
  
  # merge
  merged_diff <- base::merge(data.frame(date=lubridate::date(zoo::index(data_diff))),
                             data.frame(date = zoo::index(daily_diff), zoo::coredata(daily_diff)), all.X = TRUE, by='date')
  merged_diff <- base::cbind(zoo::coredata(data_diff[paste0(merged_diff[1, 1], '/')]), merged_diff[, -1])
  
  # indecies to remove
  indecies <- abs(merged_diff[, 1]) <  abs(merged_diff[, 5]) &
    abs(merged_diff[, 2]) <  abs(merged_diff[, 6]) & 
    abs(merged_diff[, 3]) <  abs(merged_diff[, 7]) & 
    abs(merged_diff[, 4]) <  abs(merged_diff[, 8])
  
  # remove outliers
  data <- data[indecies, ]
  
  return(data)
}

# remove outliers with defined function
stock_xts <- remove_ourlier_diff_median(stock_xts)

# usefull samples and alternative classes
stock_xts_daily <- xts::to.daily(stock_xts)
colnames(stock_xts_daily) <- c('open', 'high', 'low', 'close')
stock_df_daily <- data.frame(date = zoo::index(stock_xts_daily),
                             close = zoo::coredata(stock_xts_daily)[, 'close', drop=FALSE])

# time split
splits <- rsample::initial_time_split(stock_df_daily, prop=0.9)



# MODELING WITH MODELTIME -----

# Model 1: auto_arima
model_fit_arima_no_boost <- arima_reg() %>%
  set_engine(engine = "auto_arima") %>%
  parsnip::fit(close ~ date, data = training(splits))

# Model 2: arima_boost
model_fit_arima_boosted <- arima_boost(
  min_n = 2,
  learn_rate = 0.015
) %>%
  set_engine(engine = "auto_arima_xgboost") %>%
  fit(close ~ date + as.numeric(date),
      data = training(splits))

# Model 3: ets ----
model_fit_ets <- exp_smoothing() %>%
  set_engine(engine = "ets") %>%
  fit(close ~ date, data = training(splits))

# Model 4: prophet ----
model_fit_prophet <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(close ~ date, data = training(splits))

# Model 5: lm ----
model_fit_lm <- linear_reg() %>%
  set_engine("lm") %>%
  fit(close ~ as.numeric(date),
      data = training(splits))

# Add fitted models to a Model Table
models_tbl <- modeltime_table(
  model_fit_arima_no_boost,
  model_fit_arima_boosted,
  model_fit_ets,
  model_fit_prophet,
  model_fit_lm
)

# Calibrate the model to a testing set
calibration_tbl <- models_tbl %>%
  modeltime_calibrate(new_data = testing(splits))

# Visualize results
calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = stock_df_daily
  ) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, # For mobile screens
    .interactive      = TRUE
  )

# performances
calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(resizable = TRUE, bordered = TRUE)

# refit
refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = stock_df_daily)

refit_tbl %>%
  modeltime_forecast(h = "3 years", actual_data = stock_df_daily) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, # For mobile screens
    .interactive      = TRUE
  )


# MODELLINF WITH REMIX AUTOML ---------------------------------------------

# # convert data to data.table
# DT <- data.table::as.data.table(stock_df_daily)
# str(DT)
# DT_sample <- DT[1:100]
# 
# # AutoBanditSarima
# auto_bandit_sarima <- RemixAutoML::AutoBanditSarima(
#   data = DT_sample,
#   TargetVariableName = "close",
#   DateColumnName = "date",
#   TimeAggLevel = "day",
#   EvaluationMetric = "MAE",
#   NumHoldOutPeriods = 5L,
#   NumFCPeriods = 5L,
#   MaxLags = 5L,
#   MaxSeasonalLags = 0L,
#   MaxMovingAverages = 5L,
#   MaxSeasonalMovingAverages = 0L,
#   MaxFourierPairs = 2L,
#   TrainWeighting = 0.50,
#   MaxConsecutiveFails = 50L,
#   MaxNumberModels = 500L,
#   MaxRunTimeMinutes = 30L)

# 
# DT <- structure(list(date = structure(c(10228, 10231, 10232, 10233, 
#                                         10234, 10235, 10238, 10239, 10240, 10241, 10242, 10246, 10247, 
#                                         10248, 10249, 10252, 10253, 10254, 10255, 10256, 10259, 10260, 
#                                         10261, 10262, 10263, 10266, 10267, 10268, 10269, 10270, 10274, 
#                                         10275, 10276, 10277, 10280, 10281, 10282, 10283, 10284, 10287, 
#                                         10288, 10289, 10290, 10291, 10294, 10295, 10296, 10297, 10298, 
#                                         10301, 10302, 10303, 10304, 10305, 10308, 10309, 10310, 10311, 
#                                         10312, 10315, 10316, 10317, 10318, 10319, 10322, 10323, 10324, 
#                                         10325, 10329, 10330, 10331, 10332, 10333, 10336, 10337, 10338, 
#                                         10339, 10340, 10343, 10344, 10345, 10346, 10347, 10350, 10351, 
#                                         10352, 10353, 10354, 10357, 10358, 10359, 10360, 10361, 10364, 
#                                         10365, 10366, 10367, 10368, 10372, 10373), tclass = "Date", tzone = "UTC", class = "Date"), 
#                      close = c(97.5625, 97.78125, 96.21875, 96.46875, 95.625, 
#                                92.3125, 94, 95.3125, 95.75, 94.9375, 96.3125, 97.875, 96.9375, 
#                                96.0625, 95.9375, 95.875, 96.84375, 97.71875, 98.25, 98.3125, 
#                                99.9375, 100.6875, 100.5625, 100.5, 101.625, 101.28125, 102.25, 
#                                102.15625, 102.59375, 102, 102.5, 103.4375, 102.875, 103.65625, 
#                                104.0625, 103.25, 104.53125, 105.125, 105.125, 104.90625, 
#                                105.5, 104.8125, 103.84375, 105.9375, 105.5625, 106.5625, 
#                                107.0625, 107.5, 107.09375, 108.25, 108.5625, 108.96875, 
#                                109.25, 109.875, 109.625, 110.5625, 110.15625, 110.09375, 
#                                109.625, 109.5625, 109.9375, 110.8125, 112.03125, 112.59375, 
#                                111.6875, 110.9375, 110.3125, 111.1875, 110.875, 111.8125, 
#                                112.125, 110.8125, 112.28125, 112.25, 112.78125, 113.09375, 
#                                112, 110.8125, 108.71875, 108.5625, 109.3125, 111.34375, 
#                                112.59375, 112.3125, 111.53125, 110.21875, 109.34375, 111.125, 
#                                110.75, 111.9375, 112.21875, 111.65625, 111.03125, 110.59375, 
#                                111.34375, 112.40625, 111.6875, 111.25, 109.46875, 109.625
#                      )), class = c("data.table", "data.frame"), row.names = c(NA, 
#                                                                               -100L), .internal.selfref = <pointer: 0x0000025c0c131ef0>)
# 
# 
# Output <- RemixAutoML::AutoBanditSarima(
#    data = DT_sample,
#    TargetVariableName = "close",
#    DateColumnName = "date",
#    TimeAggLevel = "day",
#    EvaluationMetric = "MAE",
#    NumHoldOutPeriods = 5L,
#    NumFCPeriods = 5L,
#    MaxLags = 5L,
#    MaxSeasonalLags = 0L,
#    MaxMovingAverages = 5L, 
#    MaxSeasonalMovingAverages = 0L,
#    MaxFourierPairs = 2L,
#    TrainWeighting = 0.50,
#    MaxConsecutiveFails = 50L,
#    MaxNumberModels = 500L,
#    MaxRunTimeMinutes = 30L)
