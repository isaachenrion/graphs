
import torch
import torch.nn as nn

class Message(nn.Module):
    ''' Base class implementing neural message function for MPNNs. All subclasses
        should implement build_nn, which returns a lambda function
    '''
    def __init__(self, config):
        super().__init__()
        self.h_v_dim = config.hidden_dim
        self.h_w_dim = config.hidden_dim
        self.e_vw_dim = config.edge_dim
        self.message_dim = config.message_dim

    def forward(self, *args):
        pass

class FullyConnectedMessage(Message):
    def __init__(self, *args):
        super().__init__(*args)
        self.net = nn.Sequential(
            nn.Linear(self.h_v_dim + self.h_w_dim + self.e_vw_dim, self.message_dim),
            nn.ReLU()
            )

    def forward(self, *args):
        return self.net(torch.cat([*args], 1))
