import pandas as pd
from utils import get_current_season

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# DATA CLEANING
# # # # # # # # # # # # # # # # # # # # # # # # # # #


season = get_current_season()

# load collected data
df = pd.read_csv(f"backend/data/nba_all_players_game_logs_{season}.csv")

# convert GAME_DATE to datetime
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# drop irrelevant columnns
columns_to_drop = ['VIDEO_AVAILABLE', 'WL']
df = df.drop(columns=columns_to_drop)

# save cleaned version
df.to_csv("backend/output/clean_data.csv", index = False)
print("Clean data saved to output/clean_data.csv")
