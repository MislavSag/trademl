library(data.table)
library(DBI)
library(RMySQL)
library(anytime)
library(exuber)
library(PerformanceAnalytics)
library(runner)
library(dpseg)



# IMPORT DATA -------------------------------------------------------------

# connect to database
connection <- function() {
  con <- DBI::dbConnect(RMySQL::MySQL(),
                        host = "91.234.46.219",
                        port = 3306L,
                        dbname = "odvjet12_market_data_usa",
                        username = 'odvjet12_mislav',
                        password = 'Theanswer0207',
                        Trusted_Connection = "True")
}


# import ohlcv and python frac diff data
db <- connection()
contract <- dbGetQuery(db, 'SELECT date, open, high, low, close, volume FROM SPY')
invisible(dbDisconnect(db))
contract$date <- anytime::anytime(contract$date)

# convert to xts and generate lowe frequncy series
contract_xts <- xts::xts(contract[, -1], order.by=contract$date, unique=TRUE, tzone='America/Chicago')
contract_ohlcv <- contract_xts[, c('open', 'high', 'low', 'close')]
colnames(contract_ohlcv) <- c('open', 'high', 'low', 'close')
contract_daily <- xts::to.daily(contract_ohlcv)


# REMOVE OUTLIERS ---------------------------------------------------------

# diff medain growth outlier remove
median_scaler <- 20
daily_diff <- na.omit(abs(diff(contract_daily[, -5])) + 0.05) * median_scaler
daily_diff <- data.frame(date = as.Date(zoo::index(daily_diff)), zoo::coredata(daily_diff))
data_test <- na.omit(diff(contract_xts[, -5]))
data_test <- data.frame(date_time = as.POSIXct(zoo::index(data_test)), zoo::coredata(data_test))
data_test$date <- as.Date(data_test$date_time)
data_test_diff <- base::merge(data_test, daily_diff, by = 'date', all.x = TRUE, all.y = FALSE)
indexer <- abs(data_test_diff$close) < abs(data_test_diff$contract_ohlcv.Close) &
  abs(data_test_diff$open) < abs(data_test_diff$contract_ohlcv.Open) &
  abs(data_test_diff$high) < abs(data_test_diff$contract_ohlcv.High) &
  abs(data_test_diff$low) < abs(data_test_diff$contract_ohlcv.Low)
contract_xts <- contract_xts[which(indexer), ]

# convert to xts and generate lowe frequncy series
contract_ohlcv <- contract_xts[, c('open', 'high', 'low', 'close')]
colnames(contract_ohlcv) <- c('open', 'high', 'low', 'close')
contract_daily <- xts::to.daily(contract_ohlcv)
contract_hourly <- xts::to.hourly(contract_ohlcv)
contract_weekly <- xts::to.weekly(contract_ohlcv)
contract_monthly <- xts::to.monthly(contract_ohlcv)
close_daily <- contract_daily$contract_ohlcv.Close
close_hourly <- contract_hourly$contract_ohlcv.Close
close_weekly <- contract_weekly$contract_ohlcv.Close
close_monthly <- contract_monthly$contract_ohlcv.Close
price <- zoo::coredata(close_daily)
price_minute <- zoo::coredata(contract_ohlcv$close)



# TREND FOLLOWING USING LINEAR SEGENTATION --------------------------------


# parameters
freq <- 'hourly'
roll_window <- 600
type <- "var"  # use the (default) scoring, -var(residuals(lm(y~x)))
jumps <- FALSE # allow discrete jumps between segments?


# prepare data
if (freq == 'daily') {
  x <- as.numeric(zoo::index(close_daily))
  y <- zoo::coredata(close_daily)
  time_xts <- zoo::index(close_daily)
} else if (freq == 'hourly') {
  x <- as.numeric(zoo::index(close_hourly))
  y <- zoo::coredata(close_hourly)
  time_xts <- zoo::index(close_hourly)
} else if (freq == 'minute') {
  x <- as.numeric(zoo::index(contract_ohlcv$close))
  y <- zoo::coredata(contract_ohlcv$close)
  time_xts <- zoo::index(contract_ohlcv$close)
}

# apply lin segment by rolling
DT <- data.table::as.data.table(cbind(x, y))
colnames(DT) <- c('time', 'price')
# DT <- DT[5500:nrow(DT)]
dpseg_roll <- function(data) {
  p <- estimateP(x=data$time, y=data$price, plot=FALSE)
  segs <- dpseg(data$time, data$price, jumps=jumps, P=p, type=type, store.matrix=TRUE, verb=FALSE)
  slope_last <- segs$segments$slope[length(segs$segments$slope)]
  return(slope_last)
}
DT[, 
   slope := runner(
     x = .SD,
     f = dpseg_roll,
     k = 600,
     na_pad = TRUE
   )]
DT <- DT[!is.na(slope)]

# backtest
DT[, sign := ifelse(slope > 0, 1, 0)]
DT[, returns := (price - data.table::shift(price, 1L, type='lag')) / data.table::shift(price, 1L, type='lag')]
DT <- DT[-1L]
DT[, returns_strategy := returns * sign]
pref <- as.xts(cbind.data.frame(benchmarg = DT$returns, strategy = DT$returns_strategy),
               order.by = time_xts[roll_window:(roll_window+nrow(DT)-1)])
head(pref)
charts.PerformanceSummary(pref)