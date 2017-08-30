import torch
import torch.nn as nn

class VertexUpdate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.message_dim = config.message_dim
        self.vertex_state_dim = config.vertex_state_dim

    def forward(self, *args):
        pass


class GRUVertexUpdate(VertexUpdate):
    def __init__(self, config):
        super().__init__(config)
        self.gru = nn.GRUCell(self.message_dim + self.vertex_state_dim, self.hidden_dim)

    def _forward_with_vertex_state(self, m_v, h_v, x_v):
        h_v = self.gru(torch.cat([m_v, x_v], 1), h_v)
        return h_v

    def _forward_without_vertex_state(self, m_v, h_v):
        #import ipdb; ipdb.set_trace()

        h_v = self.gru(m_v, h_v)
        return h_v

    def forward(self, *args):
        if self.vertex_state_dim > 0:
            return self._forward_with_vertex_state(*args)
        else:
            return self._forward_without_vertex_state(*args)

def make_vertex_update(vertex_update_config):
    if vertex_update_config.function == 'gru':
        return GRUVertexUpdate(vertex_update_config.config)
    else:
        raise ValueError("Unsupported vertex update function! ({})".format(vertex_update_config.function))
