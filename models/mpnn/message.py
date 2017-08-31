
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
        return torch.cat([h_w, e_vw], -1)

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

class EdgeMessage(Message):
    def __init__(self, *args):
        super().__init__(*args)
        self.edge_net = nn.Linear(self.e_vw_dim, self.message_dim * self.h_w_dim)

    def forward(self, h_w, e_vw):

        if h_w.dim() == 2:
            message = torch.matmul(
                self.edge_net(e_vw).view(e_vw.size()[0], self.message_dim, self.h_w_dim),
                h_w.unsqueeze(-1)).squeeze(-1)

        elif h_w.dim() == 3:
            message = torch.matmul(
                self.edge_net(e_vw).view(e_vw.size()[0], e_vw.size()[1], self.message_dim, self.h_w_dim),
                h_w.unsqueeze(-1)).squeeze(-1)

        elif e_vw.dim() == 4:
            h_w = h_w.unsqueeze(1).expand(
                        h_w.size()[0],
                        e_vw.size()[1],
                        e_vw.size()[2],
                        self.config.hidden_dim
                        )
            message = torch.matmul(
                self.edge_net(e_vw).view(e_vw.size()[0], e_vw.size()[1], e_vw.size()[2], self.message_dim, self.h_w_dim),
                h_w.unsqueeze(-1)).squeeze(-1)
            message = torch.mean(message, 2)
        return message

class Constant(Message):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, h_w, e_vw):
        if e_vw.dim() == 3:
            h_w = h_w.unsqueeze(2).expand(e_vw.size()[0], e_vw.size()[1], h_w.size()[-1])
            message = torch.cat([e_vw, h_w], -1)
        elif e_vw.dim() == 4:
            h_w = h_w.unsqueeze(2).expand(e_vw.size()[0], e_vw.size()[1], e_vw.size()[2], h_w.size()[-1])
            message = torch.cat([e_vw, h_w], -1)
            message = torch.mean(message, 2)
        return message


def make_message(message_config):
    if message_config.function == 'fully_connected':
        return FullyConnectedMessage(message_config.config)
    elif message_config.function == 'constant':
        return Constant(message_config.config)
    elif message_config.function == 'edge_message':
        return EdgeMessage(message_config.config)
    else:
        raise ValueError("Unsupported message function! ({})".format(message_config.function))
