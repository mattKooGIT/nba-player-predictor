import pandas as pd

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# DATA CLEANING
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# load collected data
df = pd.read_csv("backend/data/nba_all_players_game_logs_2024.csv")

# convert GAME_DATE to datetime
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# drop irrelevant columnns
columns_to_drop = ['VIDEO_AVAILABLE', 'WL']
df = df.drop(columns=columns_to_drop)

# save cleaned version
df.to_csv("backend/output/clean_data.csv", index = False)
print("Clean data saved to output/clean_data.csv")