library(data.table)
library(fracdiff)
library(anytime)
library(purrr)
library(reticulate)
reticulate::use_python('C:/ProgramData/Anaconda3/python.exe', required = TRUE)
pd <- reticulate::import('pandas')



# FLAGS -------------------------------------------------------------------

input_path = 'D:/market_data/usa/ohlcv_features'


# IMPORT DATA -------------------------------------------------------------

# import big table
df <- pd$read_hdf(file.path(input_path, 'SPY_raw_ta_labels.h5'))
date <- attributes(df)
df$date <- anytime::anytime(date$row.names, tz='America/Chicago')
df <- data.table::as.data.table(df)

df_day <- df[, .SD[c(.N)], by=cut(date, "day")]
df_day <- df_day[, .SD, .SDcols=which(!grepl('fracdiff_', colnames(df_day)))]
df_day$cut <- NULL
df_day <- df_day[, .SD, .SDcols=1:which(colnames(df_day) == 'vix_close_open')]
df_day$tick_rule <- NULL
df_day$volume_vix <- NULL
df_day_colnames <- colnames(df_day)



  