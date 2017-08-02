import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


class MPNN(nn.Module):
    def __init__(self, vertex_dim, state_dim, edge_dim, message_dim, readout_dim, n_iters, config):
        super().__init__()
        self.config = config
        self.vertex_dim, self.state_dim, self.edge_dim, self.message_dim, self.readout_dim, self.n_iters = \
            vertex_dim, state_dim, edge_dim, message_dim, readout_dim, n_iters
        readout_hidden_dim = 10
        self.message = config.message_function(state_dim, state_dim, edge_dim, message_dim)
        self.vertex_update = config.vertex_update_function(state_dim, message_dim)
        self.readout = config.readout_function(state_dim, readout_hidden_dim, readout_dim)
        self.embedding = config.embedding_function(vertex_dim, state_dim)

    def forward(self, G):
        # embed the data
        for v in G.nodes():
            G.node[v]['hidden'] = self.embedding(G.node[v]['data'])

        # iterate message passing
        for idx in range(self.n_iters):
            for v in G.nodes():
                h_v = G.node[v]['hidden']
                m_v = G.node[v]['message']
                for w in G.neighbors(v):
                    h_w = G.node[w]['hidden']
                    e_vw = G.edge[v][w]['data']
                    m_vw = self.message(h_v, h_w, e_vw)
                    m_v = m_v + m_vw
                G.node[v]['message'] = m_v
                G.node[v]['hidden'] = self.vertex_update(m_v, h_v)

        # readout
        out = self.readout([G.node[v]['hidden'] for v in G.nodes()])
        return out
