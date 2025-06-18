# model_utils.py
import torch.nn as nn

# rolling features used as input
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

# features being predicted
target_features = ['PTS', 'AST', 'REB', 'STL', 'TOV', 'BLK', 'FG_PCT', 'FT_PCT', 'FG3M']

# model definition
class NBARegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NBARegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)
