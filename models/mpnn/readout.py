import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .set2set import Set2Vec

class Readout(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.classify = (self.config.mode == 'clf')
        self.hidden_dim = config.hidden_dim
        self.graph_targets = config.graph_targets
        if self.classify: assert len(self.graph_targets) == 1
        self.readout_dim = sum(target.dim for target in self.graph_targets) if self.graph_targets is not None else 0

    def forward(self, G):
        pass

class FullyConnectedReadout(Readout):
    def __init__(self, config):
        super().__init__(config)
        self.readout_hidden_dim = config.readout_hidden_dim
        self.activation = nn.LeakyReLU
        net = nn.Sequential(
                nn.Linear(self.hidden_dim, self.readout_hidden_dim),
                self.activation(),
                nn.Linear(self.readout_hidden_dim, self.readout_dim),
                )
        self.net = net

    def forward(self, G):
        h_vs = [G.node[v]['hidden'] for v in G.nodes()]
        x = torch.mean(torch.stack(h_vs, 2), 2)
        x = self.net(x)
        if self.classify:
            return {target.name: x for i, target in enumerate(self.graph_targets)}
        else:
            return {target.name: x[:, i] for i, target in enumerate(self.graph_targets)}


class SetReadout(Readout):
    def __init__(self, config):
        super().__init__(config)
        self.set2vec = Set2Vec(self.hidden_dim, self.readout_dim, config.readout_hidden_dim)

    def forward(self, G):
        h_vs = torch.stack([G.node[v]['hidden'] for v in G.nodes()], 1)
        x = self.set2vec(h_vs)

        if self.classify:
            return {target.name: x for i, target in enumerate(self.graph_targets)}
        else:
            return {target.name: x[:, i] for i, target in enumerate(self.graph_targets)}



class SelectedVerticesReadout(Readout):
    def __init__(self, config):
        super().__init__(config)
        self.module_list = nn.ModuleList()
        for target in self.graph_targets:
            self.module_list.append(nn.Linear(self.hidden_dim, self.readout_dim))

    def forward(self, G):
        h_dict = {v: G.node[v]['hidden'] for v in G.nodes()}
        out = {}
        for i, target in enumerate(self.graph_targets):
            out[target.name] = self.module_list[i](h_dict[target.name])
        return out

def make_readout(readout_config):
    if readout_config.function == 'fully_connected':
        return FullyConnectedReadout(readout_config.config)
    elif readout_config.function == 'selected_vertices':
        return SelectedVerticesReadout(readout_config.config)
    elif readout_config.function == 'set':
        return SetReadout(readout_config.config)
    else:
        raise ValueError("Unsupported readout function! ({})".format(readout_config.function))
