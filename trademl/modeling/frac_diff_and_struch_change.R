library(xts)
library(RMySQL)
library(DBI)
library(fracdiff)
library(strucchange)
library(ggplot2)



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
close_daily <- zoo::zoo(contract_daily$contract_ohlcv.Close, order.by=zoo::index(contract_daily))
close_hourly <- zoo::zoo(contract_hourly$contract_ohlcv.Close, order.by=zoo::index(contract_hourly))


# EXPLOSIVE TIME SERIES ----------------------------------------------------------------
exuber_test <- exuber::radf(zoo::coredata(close_daily), lag=0)
summary(exuber_test)
diagnostics(exuber_test)

library(MultipleBubbles)
price <- zoo::coredata(close_daily)[1:1000]
test <- MultipleBubbles::sadf_gsadf(price, adflag = 1, mflag = 1, IC = 'BIC', parallel = TRUE)

library(MultipleBubbles)
price <- c(97.5625, 97.78125, 96.21875, 96.46875, 95.625, 92.3125, 94, 
           95.3125, 95.75, 94.9375, 96.3125, 97.875, 96.9375, 96.0625, 95.9375, 
           95.875, 96.84375, 97.71875, 98.25, 98.3125, 99.9375, 100.6875, 
           100.5625, 100.5, 101.625, 101.28125, 102.25, 102.15625, 102.59375, 
           102, 102.5, 103.4375, 102.875, 103.65625, 104.0625, 103.25, 104.53125, 
           105.125, 105.125, 104.90625, 105.5, 104.8125, 103.84375, 105.9375, 
           105.5625, 106.5625, 107.0625, 107.5, 107.09375, 108.25, 108.5625, 
           108.96875, 109.25, 109.875, 109.625, 110.5625, 110.15625, 110.09375, 
           109.625, 109.5625, 109.9375, 110.8125, 112.03125, 112.59375, 
           111.6875, 110.9375, 110.3125, 111.1875, 110.875, 111.8125, 112.125, 
           110.8125, 112.28125, 112.25, 112.78125, 113.09375, 112, 110.8125, 
           108.71875, 108.5625, 109.3125, 111.34375, 112.59375, 112.3125, 
           111.53125, 110.21875, 109.34375, 111.125, 110.75, 111.9375, 112.21875, 
           111.65625, 111.03125, 110.59375, 111.34375, 112.40625, 111.6875, 
           111.25, 109.46875, 109.625)
test <- MultipleBubbles::sadf_gsadf(price, adflag = 1, mflag = 1, IC = 'BIC', parallel = FALSE)





# FRACDIFF ----------------------------------------------------------------

min_d_fdGPH <- fracdiff::fdGPH(zoo::coredata(close_hourly))
min_d_fdSperio <- fracdiff::fdSperio(zoo::coredata(close_hourly))
min_d <- mean(c(min_d_fdGPH$d, min_d_fdSperio$d))
fraddiff_series <- fracdiff::diffseries(zoo::coredata(spy_xts$close), min_d)
fraddiff_ohlc <- apply(zoo::coredata(spy_ohlcv), 2, function(x) {fracdiff::diffseries(x, min_d)})
colnames(fraddiff_ohlc) <- paste0('fracdiff_r_', colnames(fraddiff_ohlc))
security <- cbind.data.frame(contract, fraddiff_ohlc)


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
