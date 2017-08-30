
import torch
import torch.nn as nn

class Message(nn.Module):
    ''' Base class implementing neural message function for MPNNs. All subclasses
        should implement build_nn, which returns a lambda function
    '''
    def __init__(self, config):
        super().__init__()
        self.h_w_dim = config.hidden_dim
        self.e_vw_dim = config.edge_dim
        self.message_dim = config.message_dim

    def forward(self, h_w, e_vw):
        pass

    def filter_input(self, h_w, e_vw):
        if e_vw is None:
            return h_w
        elif h_w is None:
            return e_vw
        return torch.cat([h_w, e_vw], 1)

class FullyConnectedMessage(Message):
    def __init__(self, *args):
        super().__init__(*args)
        self.net = nn.Sequential(
            nn.Linear(self.h_w_dim + self.e_vw_dim, self.message_dim),
            nn.ReLU()
            )

    def forward(self, h_w, e_vw):
        x = self.filter_input(h_w, e_vw)
        return self.net(x)

class Constant(Message):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, h_w, e_vw):
        x = self.filter_input(h_w, e_vw)
        return x


def make_message(message_config):
    if message_config.function == 'fully_connected':
        return FullyConnectedMessage(message_config.config)
    elif message_config.function == 'constant':
        return Constant(message_config.config)
    else:
        raise ValueError("Unsupported message function! ({})".format(message_config.function))
