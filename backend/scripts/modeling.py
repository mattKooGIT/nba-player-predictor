import pandas as pd
import torch
import torch.nn as nn
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# Modeling
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# load data
df = pd.read_csv("backend/output/engineered_data.csv")

# define rolling average features
rolling_features = [
    'PTS_5game_avg',
    'AST_5game_avg',
    'REB_5game_avg',
    'STL_5game_avg',
    'TOV_5game_avg',
    'BLK_5game_avg',
    'FG_PCT_5game_avg',
    'FT_PCT_5game_avg',
    'FG3M_5game_avg'
]

target_features = ['PTS', 'AST', 'REB', 'STL', 'TOV', 'BLK', 'FG_PCT', 'FT_PCT', 'FG3M']

# drop rows where any rolling averages are NaN
df = df[df[rolling_features].notna().all(axis=1)]

# split into training and test sets
x = df[rolling_features].values
y = df[target_features].values

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# create dataset class
class NBAPlayerDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = NBAPlayerDataset(x_train, y_train)
test_dataset = NBAPlayerDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 32)

# define the model
class NBARegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NBARegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

model = NBARegressionModel(input_dim = len(rolling_features), output_dim = len(target_features))

# define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1} / {epochs}], Loss: {running_loss / len(train_loader):.4f}")

# evaluate the model
model.eval()
with torch.no_grad():
    total_loss = 0
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item()
    print(f"\n Test Loss: {total_loss / len(test_loader):.4f}")

# save the trained model
torch.save(model.state_dict(), "backend/models/nba_model.pt")

# save the scalers
joblib.dump(scaler_x, "backend/models/scaler_x.pkl")
joblib.dump(scaler_y, "backend/models/scaler_y.pkl")

# reload the original dataframe
df_full = pd.read_csv("backend/output/engineered_data.csv")

# only keep rows with valid rolling features
df_predict = df_full[df_full[rolling_features].notna().all(axis=1)].copy()

# scale the features
x_all = scaler_x.transform(df_predict[rolling_features].values)

# reload the model
model = NBARegressionModel(input_dim=len(rolling_features), output_dim=len(target_features))
model.load_state_dict(torch.load("backend/models/nba_model.pt"))
model.eval()

# make predictions
with torch.no_grad():
    x_tensor = torch.tensor(x_all, dtype=torch.float32)
    y_pred_scaled = model(x_tensor).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

# add predicted columns to dataframe
for i, col in enumerate(target_features):
    df_predict[f"Predicted_{col}"] = y_pred[:, i]

# save to CSV
df_predict.to_csv("backend/output/predicted_stats.csv", index=False)
print("✅ Predicted stats added and saved to: predicted_stats.csv")