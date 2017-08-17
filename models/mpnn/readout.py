import torch
import torch.nn as nn
import numpy as np

class Readout(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.readout_dim = config.readout_dim

    def forward(self, h_vs):
        pass

    def build_nn(self):
        pass

class FullyConnectedReadout(Readout):
    def __init__(self, config):
        super().__init__(config)
        self.sigmoid = (self.config.mode == 'clf')
        self.readout_hidden_dim = config.readout_hidden_dim
        self.R = self.build_nn()

    def build_nn(self):
        if self.sigmoid:
            net = nn.Sequential(
                nn.Linear(self.hidden_dim, self.readout_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.readout_hidden_dim, self.readout_dim),
                nn.Sigmoid()
            )
        else:
            net = nn.Sequential(
                nn.Linear(self.hidden_dim, self.readout_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.readout_hidden_dim, self.readout_dim)
            )
        return net

    def forward(self, h_vs):
        x_ = torch.stack(h_vs, 2)
        x = torch.sum(x_, 2)
        return self.R(x)
