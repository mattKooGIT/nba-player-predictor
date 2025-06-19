import torch
import pandas as pd
import joblib
import os
import sys
from nba_api.stats.static import players

# Add backend/ to Python path so model_utils.py can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.scripts.model_utils import NBARegressionModel, rolling_features, target_features

# Define base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_player_id(player_name):
    all_players = players.get_active_players()
    for player in all_players:
        if player_name.lower() == player['full_name'].lower():
            return player['id']
    return None

def get_headshot_url(player_id):
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

# Load scalers
scaler_x = joblib.load(os.path.join(BASE_DIR, "models", "scaler_x.pkl"))
scaler_y = joblib.load(os.path.join(BASE_DIR, "models", "scaler_y.pkl"))

# Load model
model = NBARegressionModel(input_dim=len(rolling_features), output_dim=len(target_features))
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "nba_model.pt"), map_location=torch.device('cpu')))
model.eval()

# Load data
df = pd.read_csv(os.path.join(BASE_DIR, "output", "engineered_data.csv"))

# Prediction function
def predict_player_stats(player_name):
    player_data = df[df['PLAYER_NAME'].str.lower() == player_name.lower()]
    if player_data.empty:
        return {"error": "Player not found. Please try again."}

    avg_last_5 = player_data[rolling_features].tail(5).mean().to_frame().T
    x = avg_last_5.values
    x_scaled = scaler_x.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_pred_scaled = model(x_tensor)

    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())
    prediction = {k: float(v) for k, v in zip(target_features, y_pred[0])}

    full_name = player_data['PLAYER_NAME'].iloc[-1]
    player_id = get_player_id(player_name)
    headshot_url = get_headshot_url(player_id) if player_id else None

    return {
        "player_name": full_name,
        "player_id": player_id,
        "headshot_url": headshot_url,
        "predictions": prediction
    }
