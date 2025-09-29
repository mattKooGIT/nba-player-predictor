### NBA Player Performance Predictor

A full-stack project that collects NBA player data, builds predictive models of future performance, and displays predictions in a React web app.

## Overview
This project uses Python, SQL, PyTorch, and React to predict NBA player stats such as points, assists, and rebounds. It automates data collection, feature engineering, model training, and prediction display.

## Features
- **Data Collection**: Scripts pull and clean NBA player stats.
- **Feature Engineering**: Rolling averages and other metrics created to improve prediction accuracy.
- **Modeling**: PyTorch model trained on historical stats to predict future performance.
- **Database/Export**: Predictions and cleaned data saved to CSV or SQL.
- **Frontend**: React app lets you search for a player and view predicted stat lines with a clean UI.


## How It Works
1. **Data Collection** – `data_collection.py` fetches NBA player data.
2. **Data Cleaning** – `data_cleaning.py` formats, removes nulls, and standardizes columns.
3. **Feature Engineering** – `feature_engineering.py` computes rolling averages and advanced stats.
4. **Modeling** – `modeling.py` trains a PyTorch model on engineered data, saves the model, and generates predictions into `output/predicted_stats.csv`.
5. **Frontend** – React components (`PlayerInput.jsx`, `PredictionDisplay.jsx`) query the backend/CSV to display predictions.

## Technologies Used
- **Python** (pandas, PyTorch)
- **SQL / CSV** for storage
- **React + Vite** for frontend

## Getting Started
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/nba-player-predictor.git
   cd nba-player-predictor
