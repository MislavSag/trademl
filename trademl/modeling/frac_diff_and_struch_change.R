


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
contract <- dbGetQuery(db, 'SELECT date, open, high, low, close, volume FROM SPY_IB')
invisible(dbDisconnect(db))
contract$date <- anytime::anytime(contract$date)

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


# FRACDIFF ----------------------------------------------------------------

min_d_fdGPH <- fracdiff::fdGPH(zoo::coredata(close_hourly))
min_d_fdSperio <- fracdiff::fdSperio(zoo::coredata(close_hourly))
min_d <- mean(c(min_d_fdGPH$d, min_d_fdSperio$d))
fraddiff_series <- fracdiff::diffseries(zoo::coredata(spy_xts$close), min_d)
fraddiff_ohlc <- apply(zoo::coredata(spy_ohlcv), 2, function(x) {fracdiff::diffseries(x, min_d)})
colnames(fraddiff_ohlc) <- paste0('fracdiff_r_', colnames(fraddiff_ohlc))
security <- cbind.data.frame(contract, fraddiff_ohlc)



df <- pd$read_pickle('D:/algo_trading_files/X_test.pkl')



# EXPLOSIVE TIME SERIES ----------------------------------------------------------------

# standardized close price
price <- log(zoo::coredata(close_weekly))
roll_std <- zoo::rollapply(close_weekly, width = 4, FUN = sd)
stand_price <- price/roll_std
plot(stand_price)

# SUP SADF
price <- log(zoo::coredata(close_weekly))
start_time <- Sys.time()
test <- MultipleBubbles::sadf_gsadf(price, adflag = 1, mflag = 1, IC = 2, parallel = TRUE)
end_time <- Sys.time()
end_time - start_time
plot(test$badfs)
plot(test$bsadfs)
price_index <- length(price) - length(test$badfs)
sadf_spy <- cbind.data.frame(price[(price_index+1):length(price)], test$badfs)
colnames(sadf_spy) <- c('price', 'sadf')







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
