# confing
GUILD_HOME = "guild-env"
DELETE_RUNS_ON_INIT = True

# import packages
import guild.ipy as guild
import os


# Initialize Guild Home
if not os.path.exists(GUILD_HOME):
    os.mkdir("guild-env")
    
guild.set_guild_home("guild-env")

# delete runs if exist
if DELETE_RUNS_ON_INIT:
    deleted = guild.runs().delete(permanent=True)
    print("Deleted %i run(s)" % len(deleted))
    
    
_ = guild.run(train_rf.py)

print(return_val)

guild.

run.dir

# df runs
runs = guild.runs() 

