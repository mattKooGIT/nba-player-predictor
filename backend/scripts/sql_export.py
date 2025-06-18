import pandas as pd
import sqlite3

df = pd.read_csv("backend/output/engineered_data.csv")

# connect and export
conn = sqlite3.connect("NBA_Player_Stats.db")
df.to_sql("player_stats", conn, if_exists = 'replace', index = False)
conn.close()