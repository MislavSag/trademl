import pandas as pd
import guild.ipy as guild
import os
import seaborn as sns


### PANDAS OPTIONS
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# set home
# GUILD_HOME = 'C:/ProgramData/Anaconda3/.guild'
# guild.set_guild_home(GUILD_HOME)

# RUNS
def get_runs(operation):
    runs = guild.runs(operations=operation)
    runs = runs.compare()
    return runs


def clean_runs(runs, filter_metric=['accuracy_test'], good_performance=False):
    runs = runs.dropna(axis=1, how='all')
    runs = runs.sort_values(by=filter_metric, ascending=False)
    if good_performance:
        runs = runs.dropna(subset=filter_metric)
    return runs


runs = get_runs(operation=['lightgbm'])
runs_clean = clean_runs(runs)
runs_clean.head()


# HYPERPARAMETER ANALYSIS
plt.clf()
scatter_plot = sns.scatterplot(data=runs_clean[['max_depth', 'accuracy_train']], x='max_depth', y='accuracy_train')
fig = scatter_plot.get_figure()
fig.savefig('test.png')


plt.clf()
scatter_plot = sns.scatterplot(data=runs_clean[['max_depth', 'accuracy_test']], x='max_depth', y='accuracy_test')
fig = scatter_plot.get_figure()
fig.savefig('max_depth_accuracy_test.png')

# # PLAYING WITH GOOGLE NEWS

# from pygooglenews import GoogleNews

# gn = GoogleNews(lang='hr', country='HR')

# top = gn.top_news()

# business = gn.topic_headlines('business')

# # search for the best matching articles that mention MSFT and 
# # do not mention AAPL (over the past 6 month
# search = gn.search('MSFT -APPL', when = '6m')
# len(search)
