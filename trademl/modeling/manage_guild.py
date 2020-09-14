import pandas as pd
import guild.ipy as guild
import os


### PANDAS OPTIONS
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# set home
# GUILD_HOME = 'C:/ProgramData/Anaconda3/.guild'
# guild.set_guild_home(GUILD_HOME)

# df runs
runs_rf = guild.runs(operations=["pipeline-rf-opt"])
runs_rf_compare = runs_rf.compare()
runs_lstm = guild.runs(operations=["lstm"])
runs_lstm_compare = runs_lstm.compare()



runs_clean = runs_lstm_compare.dropna(axis=1, how='all')
runs_clean = runs_clean.sort_values(by=['accuracy_test'], ascending=False)
runs_clean.head()



def clean_runs(runs, filter_metric=['accuracy_test']):
    runs_clean = runs.dropna(axis=1, how='all')
    runs_clean = runs_clean.sort_values(by=filter_metric)
    runs_clean = runs_clean.dropna(subset=filter_metric)
    runs_clean = runs_clean.loc[runs_clean[filter_metric[0]] > 0.55]


runs_rf_clean = clean_runs(runs_rf_compare)
runs_lstm_compare = clean_runs(runs_lstm_compare, ['val_accuracy'])



# # PLAYING WITH GOOGLE NEWS

# from pygooglenews import GoogleNews

# gn = GoogleNews(lang='hr', country='HR')

# top = gn.top_news()

# business = gn.topic_headlines('business')

# # search for the best matching articles that mention MSFT and 
# # do not mention AAPL (over the past 6 month
# search = gn.search('MSFT -APPL', when = '6m')
# len(search)
