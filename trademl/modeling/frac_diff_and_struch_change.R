library(DBI)
library(RMySQL)
library(anytime)
library(exuber)
library(psymonitor)
library(PerformanceAnalytics)


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
contract_hourly <- xts::to.hourly(contract_ohlcv)
contract_weekly <- xts::to.weekly(contract_ohlcv)
contract_monthly <- xts::to.monthly(contract_ohlcv)
close_daily <- contract_daily$contract_ohlcv.Close
close_hourly <- contract_hourly$contract_ohlcv.Close
close_weekly <- contract_weekly$contract_ohlcv.Close
close_monthly <- contract_monthly$contract_ohlcv.Close


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


# FRAC DIFF -------------------------------------------------------------------------


find_min_d <- function(x) {
  min_d_fdGPH <- fracdiff::fdGPH(x)
  min_d_fdSperio <- fracdiff::fdSperio(x)
  min_d <- mean(c(min_d_fdGPH$d, min_d_fdSperio$d))
  return(min_d)
}
min_d <- find_min_d(price)
if (min_d > 0) {
  x <- fracdiff::diffseries(price_minute, min_d)
}

min_d <- lapply(df_day, find_min_d)
df_sample <- df[, ..df_day_colnames]
df_sample <- map2_dfr(df_sample, min_d, ~ fracdiff::diffseries(.x, .y))
head(df_sample)

colnames(fraddiff_ohlc) <- paste0('fracdiff_r_', colnames(fraddiff_ohlc))
security <- cbind.data.frame(contract, fraddiff_ohlc)

# convert to xts and generate lowe frequncy series
contract_xts <- xts::xts(contract[, -1], order.by=contract$date, unique=TRUE, tzone='America/Chicago')
contract_ohlcv <- contract_xts[, c('open', 'high', 'low', 'close')]
colnames(contract_ohlcv) <- c('open', 'high', 'low', 'close')
contract_daily <- xts::to.daily(contract_ohlcv)
contract_hourly <- xts::to.hourly(contract_ohlcv)
contract_weekly <- xts::to.weekly(contract_ohlcv)
close_daily <- contract_daily$contract_ohlcv.Close
close_hourly <- contract_hourly$contract_ohlcv.Close
close_weekly <- contract_weekly$contract_ohlcv.Close


# EXPLOSIVE TIME SERIES ----------------------------------------------------------------

# EXUBER PACKAGE
DT <- data.table::as.data.table(price)

radf_buble_last <- function(x, adf_lag=3L) {
  rsim_data <- radf(x, minw= 50, lag = adf_lag)
  bubble <- tail(rsim_data$bsadf, 1) - radf_crit$n10$gsadf_cv['90%']
  return(bubble)
  }

rolling_radf <- data.table::frollapply(DT, 600, radf_buble_last, adf_lag = 2, fill=NA, align=c("right", "left", "center"))

# save/load
write.csv(rolling_radf[[1]], 'D:/algo_trading_files/exuber/rolling_radf.csv')
rolling_radf <- read.csv('D:/algo_trading_files/exuber/rolling_radf.csv', row.names = FALSE)
rolling_radf <- rolling_radf[, 2]


# some tests
min(test[[1]], na.rm = TRUE)
sum(test[[1]] >= 0, na.rm = TRUE)
length(which(test[[1]] >= 0))

# backtest
R <- as.data.frame((price - data.table::shift(price)) / price)
R <- cbind.data.frame(R, rolling_radf)
R <- na.omit(R)
colnames(R) <- c('return', 'rolling_radf')
R$position <- ifelse(R$rolling_radf >= 0, 0, 1)
table(R$position)
R$return_strategy <- R$position * R$return
perf <- R[, c('return', 'return_strategy')]
time_index <- close_hourly[(nrow(close_hourly)-nrow(R) + 1):nrow(close_hourly), ]
time_index <- zoo::index(time_index)
perf <- xts::xts(perf, order.by = time_index)
charts.PerformanceSummary(perf)

length(time_index)
nrow(close_hourly)
nrow(perf)
head(R)
tail(R, 1000)
tail(perf, 1000)
head(perf)
tail(perf, 200)

# SUP SADF
start_time <- Sys.time()
test <- MultipleBubbles::sadf_gsadf(price, adflag = 1, mflag = 1, IC = 2, parallel = TRUE)
end_time <- Sys.time()
end_time - start_time
plot(test$badfs)
plot(test$bsadfs)
price_index <- length(price) - length(test$badfs)
sadf_spy <- cbind.data.frame(price[(price_index+1):length(price)], test$badfs)
colnames(sadf_spy) <- c('price', 'sadf')

# PSYMONITOR PACKAGE
swindow0 <- floor(length(price_sample) * (0.01 + 1.8 / sqrt(length(price_sample)))) # set minimal window size
bsadf <- PSY(price_sample, swindow0 = swindow0, IC = 2, adflag = 2)
plot(bsadf)
Tb       <- 12*2 + swindow0 - 1  # Set the control sample size
quantilesBsadf <- cvPSYwmboot(price_sample, swindow0 = swindow0, IC = 2,
                              adflag = 2, Tb = Tb, nboot = 99,
                              nCores = 2) # simulate critical values via wild bootstrap. Note that the number of cores is arbitrarily set to 2.



# BACKTEST ----------------------------------------------------------------

# Step 1: Load libraries and data
library(quantmod)
library(PerformanceAnalytics)
## 
## Attaching package: 'PerformanceAnalytics'
## The following object is masked from 'package:graphics':
## 
##     legend

getSymbols('NFCI', src = 'FRED', , from = '2000-01-01')
## [1] "NFCI"

NFCI <- na.omit(lag(NFCI)) # we can only act on the signal after release, i.e. the next day
getSymbols("^GSPC", from = '2000-01-01')
## [1] "^GSPC"

data <- na.omit(merge(NFCI, GSPC)) # merge before (!) calculating returns)
data$GSPC <- na.omit(ROC(Cl(GSPC))) # calculate returns of closing prices

# Step 2: Create your indicator
data$sig <- ifelse(data$NFCI < 1, 1, 0)
data$sig <- na.locf(data$sig)

# Step 3: Use indicator to create equity curve
perf <- na.omit(merge(data$sig * data$GSPC, data$GSPC))
colnames(perf) <- c("Stress-based strategy", "SP500")

# Step 4: Evaluate strategy performance
table.DownsideRisk(perf)
table.Stats(perf)
charts.PerformanceSummary(perf)
chart.RelativePerformance(perf[ , 1], perf[ , 2])
chart.RiskReturnScatter(perf)


# STRUCTURAL BREAKS AND EXPLOSIVE TS -------------------------------------------------------

# prepare close
fracdiff_close <- xts::xts(security[, colnames(fraddiff_ohlc)],
                           order.by=security$date, unique=TRUE, tzone='America/Chicago')
fracdiff_daily <- xts::to.daily(fracdiff_close)
colnames(fracdiff_daily) <- colnames(fraddiff_ohlc)
fracdiff_daily_close <- fracdiff_daily$fracdiff_r_close
head(fracdiff_daily)


### F TEST based on Andrews D.W.K. (1993). It can identify only one breakpoint

# test if visualization is right with default visualizatin from the package
fs_close <- Fstats(zoo::coredata(fracdiff_daily_close) ~ 1)
bp <- breakpoints(fs_close)  # breakpoints
bfac <- breakfactor(bp, breaks = length(bp$breakpoints))  # Generates a factor encoding the segmentation given by a set of breakpoints
table(bfac)  # how many observations in each subject
fm <- lm(fracdiff_daily_close ~ bfac + 1)  # estimate regression for bot regimes

# ggplot visualization
close_df <- tsbox::ts_data.frame(fracdiff_daily)
ggplot(data=close_df, aes(x=time, y=log(value))) +
  geom_line() +
  geom_vline(xintercept=close_df[bp$breakpoints, 'time'], linetype='dotted', color='red', size=0.9) +
  geom_line(data=tsbox::ts_data.frame(zoo(predict(fm), order.by=index(fracdiff_daily_close))), aes(time, value),
            color='darkgreen', size=0.9) +
  theme_bw()

chow_test <- function(x) {
  # test if visualization is right with default visualizatin from the package
  fs_close <- Fstats(zoo::coredata(x) ~ 1)
  bp <- breakpoints(fs_close)  # breakpoints
  bfac <- breakfactor(bp, breaks = length(bp$breakpoints))  # Generates a factor encoding the segmentation given by a set of breakpoints
  table(bfac)  # how many observations in each subject
  fm <- lm(x ~ bfac + 1)  # estimate regression for bot regimes

  # ggplot visualization
  close_df <- tsbox::ts_data.frame(fracdiff_daily)
  p <- ggplot(data=close_df, aes(x=time, y=value)) +
    geom_line() +
    geom_vline(xintercept=close_df[bp$breakpoints, 'time'], linetype='dotted', color='red', size=0.9) +
    geom_line(data=tsbox::ts_data.frame(zoo(predict(fm), order.by=index(x))), aes(time, value),
              color='darkgreen', size=0.9) +
    theme_bw()

  return(p)
}

p <- chow_test(fracdiff_daily_close)
p <- chow_test()
