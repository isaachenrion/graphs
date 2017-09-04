import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple

from .readout import make_readout
from .message import make_message
from .embedding import make_embedding
from .vertex_update import make_vertex_update

class BaseMPNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_iters = config.n_iters
        self.message = make_message(config.message)
        self.vertex_update = make_vertex_update(config.vertex_update)
        self.readout = make_readout(config.readout)

    def forward(self, G):
        self.embed_data(G)
        for i in range(self.n_iters):
            self.message_passing(G)
        out = self.readout(G)
        return out

    def message_passing(self, G):
        if self.config.parallelism == 0:
            return self._message_passing_serial(G)

        elif self.config.parallelism == 1:
            return self._message_passing_edge_parallel(G)

        else:
            raise ValueError("Unsupported parallelism level!")

    def embed_data(self, G):
        pass

    def reset_hidden_states(self, G):
        try:
            batch_size = G.graph['batch_size']
        except KeyError:
            batch_size = 1
        for u in G.nodes():
            h = Variable(torch.zeros(batch_size, self.config.message.config.hidden_dim))
            if torch.cuda.is_available():
                h = h.cuda()
            G.node[u]['hidden'] = h

        return None

class VertexOnlyMPNN(BaseMPNN):
    def __init__(self, config):
        super().__init__(config)
        self.embedding = make_embedding(config.embedding)

    def _message_passing_serial(self, G):

        for v in G.nodes():
            if len(G.neighbors(v)) > 0:
                x_v = G.node[v]['state']
                h_v = G.node[v]['hidden']
                m_vws = []
                for w in G.neighbors(v):
                    h_w = G.node[w]['hidden']
                    m_vws.append(self.message(h_w))
                m_v = torch.mean(torch.stack(m_vws, 2), 2)
                G.node[v]['hidden'] = self.vertex_update(m_v, h_v, x_v)
        return None

    def _message_passing_edge_parallel(self, G):

        for v in G.nodes():
            if len(G.neighbors(v)) > 0:
                x_v = G.node[v]['state']
                h_v = G.node[v]['hidden']
                m_vws = []
                h_ws = torch.stack([G.node[w]['hidden'] for w in G.neighbors(v)], 1)
                m_vws = self.message(h_ws)
                m_v = torch.mean(m_vws, 1)
                G.node[v]['hidden'] = self.vertex_update(m_v, h_v, x_v)

        return None

    def embed_data(self, G):
        for v in G.nodes():
            G.node[v]['state'] = self.embedding(G.node[v]['data'])
        return None

class GeneralMPNN(BaseMPNN):
    def __init__(self, config):
        super().__init__(config)
        self.embedding = make_embedding(config.embedding)

    def embed_data(self, G):
        for v in G.nodes():
            G.node[v]['state'] = self.embedding(G.node[v]['data'])
        return None

    def _message_passing_serial(self, G):

        for v in G.nodes():
            if len(G.neighbors(v)) > 0:
                x_v = G.node[v]['state']
                h_v = G.node[v]['hidden']
                m_vws = []
                for w in G.neighbors(v):
                    h_w = G.node[w]['hidden']
                    e_vw = G.edge[v][w]['data']
                    m_vws.append(self.message(h_w, e_vw))
                m_v = torch.mean(torch.stack(m_vws, 2), 2)
                G.node[v]['hidden'] = self.vertex_update(m_v, h_v, x_v)

        return None

    def _message_passing_edge_parallel(self, G):

        for v in G.nodes():
            if len(G.neighbors(v)) > 0:
                x_v = G.node[v]['state']
                h_v = G.node[v]['hidden']
                m_vws = []
                h_ws = torch.stack([G.node[w]['hidden'] for w in G.neighbors(v)], 1)
                e_vws = torch.stack([G.edge[v][w]['data'] for w in G.neighbors(v)], 1)
                m_vws = self.message(h_ws, e_vws)
                m_v = torch.mean(m_vws, 1)
                G.node[v]['hidden'] = self.vertex_update(m_v, h_v, x_v)

        return None


class EdgeOnlyMPNN(BaseMPNN):
    def __init__(self, config):
        super().__init__(config)

    def _message_passing_serial(self, G):
        for v in G.nodes():
            if len(G.neighbors(v)) > 0:
                h_v = G.node[v]['hidden']
                m_vws = []
                for w in G.neighbors(v):
                    h_w = G.node[w]['hidden']
                    e_vw = G.edge[v][w]['data']
                    m_vws.append(self.message(h_w, e_vw))
                m_v = torch.mean(torch.stack(m_vws, 2), 2)
                G.node[v]['hidden'] = self.vertex_update(m_v, h_v)

        return None

    def _message_passing_edge_parallel(self, G):
        for v in G.nodes():
            if len(G.neighbors(v)) > 0:
                h_v = G.node[v]['hidden']
                m_vws = []
                h_ws = torch.stack([G.node[w]['hidden'] for w in G.neighbors(v)], 1)
                e_vws = torch.stack([G.edge[v][w]['data'] for w in G.neighbors(v)], 1)
                m_vws = self.message(h_ws, e_vws)
                m_v = torch.mean(m_vws, 1)
                G.node[v]['hidden'] = self.vertex_update(m_v, h_v)

        return None

class StructureOnlyMPNN(BaseMPNN):
    def __init__(self, config):
        super().__init__(config)

    def _message_passing_serial(self, G):
        for v in G.nodes():
            if len(G.neighbors(v)) > 0:
                h_v = G.node[v]['hidden']
                m_vws = []
                for w in G.neighbors(v):
                    h_w = G.node[w]['hidden']
                    m_vws.append(self.message(h_w))
                m_v = torch.mean(torch.stack(m_vws, 2), 2)
                G.node[v]['hidden'] = self.vertex_update(m_v, h_v)
        return None

    def _message_passing_edge_parallel(self, G):
        for v in G.nodes():
            if len(G.neighbors(v)) > 0:
                h_v = G.node[v]['hidden']
                m_vws = []
                h_ws = torch.stack([G.node[w]['hidden'] for w in G.neighbors(v)], 1)
                m_vws = self.message(h_ws)
                m_v = torch.mean(m_vws, 1)
                G.node[v]['hidden'] = self.vertex_update(m_v, h_v)

        return None


def make_mpnn(mpnn_config):
    if mpnn_config.embedding.config.data_dim == 0:
        if mpnn_config.message.config.edge_dim == 0:
            return StructureOnlyMPNN(mpnn_config)
        return EdgeOnlyMPNN(mpnn_config)
    elif mpnn_config.message.config.edge_dim == 0:
        return VertexOnlyMPNN(mpnn_config)
    return GeneralMPNN(mpnn_config)
