from flask import Flask, request, jsonify
from .predict import predict_player_stats
from flask_cors import CORS
import pandas as pd
import os
from backend.model_utils import NBARegressionModel
import torch
import joblib

# Define constants
rolling_features = [
    'PTS_5game_avg', 'AST_5game_avg', 'REB_5game_avg',
    'STL_5game_avg', 'TOV_5game_avg', 'BLK_5game_avg',
    'FG_PCT_5game_avg', 'FT_PCT_5game_avg', 'FG3M_5game_avg'
]
target_features = ['PTS', 'AST', 'REB', 'STL', 'TOV', 'BLK', 'FG_PCT', 'FT_PCT', 'FG3M']

# Load model and scalers once
model = NBARegressionModel(input_dim=len(rolling_features), output_dim=len(target_features))
model.load_state_dict(torch.load("backend/models/nba_model.pt"))
model.eval()

scaler_x = joblib.load("backend/models/scaler_x.pkl")
scaler_y = joblib.load("backend/models/scaler_y.pkl")


app = Flask(__name__)
CORS(app, origins="https://nba-player-predictor.vercel.app")

# load once when app starts to get dynamic playerbase
df = pd.read_csv("backend/output/predicted_stats.csv")

@app.route('/players', methods = ['GET'])
def get_players():
    if df.empty or "PLAYER_NAME" not in df.columns:
        return jsonify({"error": "Player data not loaded."}), 500
    
    player_names = df["PLAYER_NAME"].dropna().unique().tolist()
    return jsonify({"players": player_names})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    player_name = data.get('player_name')

    if not player_name:
        return jsonify({"error": "Please enter a player name."}), 400
    prediction = predict_player_stats(player_name)
    if 'error' in prediction:
        return jsonify(prediction), 404
    
    return jsonify(prediction)

if __name__ == "__main__":
    # Use 0.0.0.0 so it listens externally on Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
