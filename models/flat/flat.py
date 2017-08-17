import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


class Flat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.state_dim, self.hidden_dim, self.readout_dim = config.state_dim, config.hidden_dim, config.readout_dim
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.readout_dim)

    def forward(self, G):
        flat_graph_state = G.graph['flat_graph_state']
        x = F.relu(self.fc1(flat_graph_state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.mode == 'clf':
            x = F.sigmoid(x)
        return x

    def reset_hidden_states(self, G):
        return G

def make_flat(config):
    return Flat(config)
