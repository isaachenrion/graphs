
import torch
import torch.nn as nn

class Message(nn.Module):
    ''' Base class implementing neural message function for MPNNs. All subclasses
        should implement build_nn, which returns a lambda function
    '''
    def __init__(self, h_v_dim, h_w_dim, e_vw_dim, message_dim):
        super().__init__()
        self.h_v_dim = h_v_dim
        self.h_w_dim = h_w_dim
        self.e_vw_dim = e_vw_dim
        self.message_dim = message_dim


    def forward(self, h_v, h_w, e_vw):
        return self.M(h_v, h_w, e_vw)

    def build_nn(self):
        pass

class FullyConnectedMessage(Message):
    def __init__(self, *args):
        super().__init__(*args)
        self.M = self.build_nn()

    def build_nn(self):
        net = nn.Sequential(
            nn.Linear(self.h_v_dim + self.h_w_dim + self.e_vw_dim, self.message_dim),
            nn.ReLU()
            )
        return net
        
    def forward(self, h_v, h_w, e_vw):
        return self.M(torch.cat([h_v, h_w, e_vw], 1))
