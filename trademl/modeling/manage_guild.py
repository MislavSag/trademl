import guild.ipy as guild
import os

# set home
GUILD_HOME = 'C:/ProgramData/Anaconda3/.guild'
guild.set_guild_home()

# df runs
runs = guild.runs() 
runs.info()

# scalars
scalars = runs.scalars()

# compare
guild.runs().compare()
