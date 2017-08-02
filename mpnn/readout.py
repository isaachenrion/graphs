import torch
import torch.nn as nn
import numpy as np

class Readout(nn.Module):
    def __init__(self, state_dim, hidden_dim, readout_dim):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.readout_dim = readout_dim

    def forward(self, h_vs):
        pass

    def build_nn(self):
        pass

class FullyConnectedReadout(Readout):
    def __init__(self, *args):
        super().__init__(*args)
        self.R = self.build_nn()

    def build_nn(self):
        net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.readout_dim),
            nn.Sigmoid()
        )
        return net

    def forward(self, h_vs):
        x_ = torch.stack(h_vs, 2)
        x = torch.sum(x_, 2).squeeze(2)
        return self.R(x)
