# confing
GUILD_HOME = "guild-rf"
DELETE_RUNS_ON_INIT = True

# import packages
import guild.ipy as guild
import os


# Initialize Guild Home
if not os.path.exists(GUILD_HOME):
    os.mkdir(GUILD_HOME)
    
guild.set_guild_home(GUILD_HOME)

# HERE RUN MY TRAIN SAVED IN SUBMODUL (LOOK AT ABOVE GUILD YML)
_ = guild.run(train)

print(return_val)

guild.

run.dir

# df runs
runs = guild.runs() 

