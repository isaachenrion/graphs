import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple



class BaseMPNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_iters = config.n_iters
        self.message = config.message.function(config.message.config)
        self.vertex_update = config.vertex_update.function(config.vertex_update.config)
        self.readout = config.readout.function(config.readout.config)

    def forward(self, G):
        pass

    def reset_hidden_states(self, G):
        pass

class VertexOnlyMPNN(BaseMPNN):
    def __init__(self, config):
        super().__init__(config)
        self.embedding = config.embedding.function(config.embedding.config)

    def forward(self, G):
        # embed the data
        for v in G.nodes():
            G.node[v]['state'] = self.embedding(G.node[v]['data'])

        # iterate message passing
        for idx in range(self.n_iters):
            for v in G.nodes():
                x_v = G.node[v]['state']
                h_v = G.node[v]['hidden']
                m_v = G.node[v]['message']
                for w in G.neighbors(v):
                    h_w = G.node[w]['hidden']
                    m_vw = self.message(h_v, h_w)
                    m_v = m_v + m_vw
                G.node[v]['message'] = m_v
                G.node[v]['hidden'] = self.vertex_update(m_v, h_v, x_v)

        # readout
        out = self.readout([G.node[v]['hidden'] for v in G.nodes()])
        return out

    def reset_hidden_states(self, G):
        for u in G.nodes():
            G.node[u]['hidden'] = Variable(torch.zeros(1, self.config.message.config.hidden_dim))
            G.node[u]['message'] = Variable(torch.zeros(1, self.config.message.config.message_dim))
        return G

class GeneralMPNN(BaseMPNN):
    def __init__(self, config):
        super().__init__(config)
        self.embedding = config.embedding.function(config.embedding[1])

    def forward(self, G):
        # embed the data
        for v in G.nodes():
            G.node[v]['state'] = self.embedding(G.node[v]['data'])

        # iterate message passing
        for idx in range(self.n_iters):
            for v in G.nodes():
                x_v = G.node[v]['state']
                h_v = G.node[v]['hidden']
                m_v = G.node[v]['message']
                for w in G.neighbors(v):
                    h_w = G.node[w]['hidden']
                    e_vw = G.edge[v][w]['edge']
                    m_vw = self.message(h_v, h_w, e_vw)
                    m_v = m_v + m_vw
                G.node[v]['message'] = m_v
                G.node[v]['hidden'] = self.vertex_update(m_v, h_v, x_v)

        # readout
        out = self.readout([G.node[v]['hidden'] for v in G.nodes()])
        return out

    def reset_hidden_states(self, G):
        for u in G.nodes():
            G.node[u]['hidden'] = Variable(torch.zeros(1, self.config.message.config.hidden_dim))
            G.node[u]['message'] = Variable(torch.zeros(1, self.config.message.config.message_dim))
        return G

class EdgeOnlyMPNN(BaseMPNN):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, G):
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

    def reset_hidden_states(self, G):
        batch_size = G.graph['readout'].size()[0]
        for u in G.nodes():
            G.node[u]['hidden'] = Variable(torch.zeros(batch_size, self.config.message.config.hidden_dim))
            G.node[u]['message'] = Variable(torch.zeros(batch_size, self.config.message.config.message_dim))
        return G

def make_mpnn(config):
    if config.vertex_update.config.vertex_state_dim == 0:
        return EdgeOnlyMPNN(config)
    elif config.message.config.edge_dim == 0:
        return VertexOnlyMPNN(config)
    else:
        return GeneralMPNN(config)
