from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import time

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# DATA COLLECTION
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# get a list of all active NBA players
all_players = players.get_active_players()
print(f"Found {len(all_players)} active players.")

all_data = []

for player in all_players:
    try:
        name = player['full_name']
        player_id = player['id']
        gamelogs = playergamelog.PlayerGameLog(
            player_id = player_id,
            season = '2024'
        )
        df = gamelogs.get_data_frames()[0]
        df['PLAYER_NAME'] = name
        all_data.append(df)

        print(f"Collected: {name}")
        time.sleep(1)
    except Exception as e:
        print(f"Skipped {name}: {e}")
        continue

# combine all players' game logs
final_df = pd.concat(all_data)

# save to CSV
final_df.to_csv("nba_all_players_game_logs_2024.csv", index = False)