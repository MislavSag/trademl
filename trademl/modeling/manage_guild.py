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
# runs = guild.runs() 
#runs.info()

# scalars
# scalars = runs.scalars()

# compare
runs_compare = guild.runs().compare()
runs_compare.head(15)
runs_compare.sort_values(by=['mean_score'], ascending=False).head()
runs_compare.sort_values(by=['fi_accuracy_test'], ascending=False).head()
runs_compare.sort_values(by=['accuracy_test'], ascending=False).head()
runs_compare.sort_values(by=['mean_score'], ascending=False).head(1)['label'].iloc[0]


# # PLAYING WITH GOOGLE NEWS

# from pygooglenews import GoogleNews

# gn = GoogleNews(lang='hr', country='HR')

# top = gn.top_news()

# business = gn.topic_headlines('business')

# # search for the best matching articles that mention MSFT and 
# # do not mention AAPL (over the past 6 month
# search = gn.search('MSFT -APPL', when = '6m')
# len(search)
