import pandas as pd

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# FEATURE ENGINEERING
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# load clean data
df = pd.read_csv("backend/output/clean_data.csv")

# sort by player and data to do time-based operations
df = df.sort_values(['PLAYER_NAME', 'GAME_DATE'])

# rolling averages
df['FG_PCT_5game_avg'] = df.groupby('PLAYER_NAME')['FG_PCT'].transform(lambda x: x.rolling(5).mean())
df['FT_PCT_5game_avg'] = df.groupby('PLAYER_NAME')['FT_PCT'].transform(lambda x: x.rolling(5).mean())
df['FG3M_5game_avg'] = df.groupby('PLAYER_NAME')['FG3M'].transform(lambda x: x.rolling(5).mean())
df['PTS_5game_avg'] = df.groupby('PLAYER_NAME')['PTS'].transform(lambda x: x.rolling(5).mean())
df['AST_5game_avg'] = df.groupby('PLAYER_NAME')['AST'].transform(lambda x: x.rolling(5).mean())
df['REB_5game_avg'] = df.groupby('PLAYER_NAME')['REB'].transform(lambda x: x.rolling(5).mean())
df['STL_5game_avg'] = df.groupby('PLAYER_NAME')['STL'].transform(lambda x: x.rolling(5).mean())
df['BLK_5game_avg'] = df.groupby('PLAYER_NAME')['BLK'].transform(lambda x: x.rolling(5).mean())
df['TOV_5game_avg'] = df.groupby('PLAYER_NAME')['TOV'].transform(lambda x: x.rolling(5).mean())

# days of rest
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df['PREV_GAME_DATE'] = df.groupby('PLAYER_NAME')['GAME_DATE'].shift(1)
df['PREV_GAME_DATE'] = pd.to_datetime(df['PREV_GAME_DATE'])
df['DAYS_REST'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days.fillna(0)

# save output
df.to_csv("backend/output/engineered_data.csv", index = False)
print("Feature-engineered data saved to output/engineered_data.csv")

