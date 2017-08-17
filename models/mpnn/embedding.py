import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dim = config.data_dim
        self.state_dim = config.state_dim

    def forward(self, x):
        pass

class FullyConnectedEmbedding(Embedding):
    def __init__(self, *args):
        super().__init__(*args)
        self.net = nn.Sequential(
            nn.Linear(self.data_dim, self.state_dim),
            nn.ReLU(),
            nn.Linear(self.state_dim, self.state_dim)
        )

    def forward(self, x):
        return self.net(x)
