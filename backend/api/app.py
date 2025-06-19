from flask import Flask, request, jsonify
from predict import predict_player_stats
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# load once when app starts to get dynamic playerbase
df = pd.read_csv("output/engineered_data.csv")

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
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)
