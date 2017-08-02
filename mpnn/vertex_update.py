import torch
import torch.nn as nn
class VertexUpdate(nn.Module):
    def __init__(self, state_dim, message_dim):
        super().__init__()
        self.state_dim = state_dim
        self.message_dim = message_dim
        self.U = self.build_nn()

    def forward(self, m_v, h_v):
        return self.U(m_v, h_v)

    def build_nn(self):
        pass

class GRUVertexUpdate(VertexUpdate):
    def __init__(self, *args):
        super().__init__(*args)

    def build_nn(self):
        net = nn.GRUCell(self.message_dim, self.state_dim)
        return net
