import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dim = config.data_dim
        self.state_dim = config.state_dim

    def forward(self, x):
        pass

class FullyConnectedEmbedding(Embedding):
    def __init__(self, *args):
        super().__init__(*args)
        self.net = nn.Sequential(
            nn.Linear(self.data_dim, self.state_dim),
            nn.ReLU(),
            nn.Linear(self.state_dim, self.state_dim)
        )

    def forward(self, x):
        return self.net(x)

class Constant(Embedding):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):
        return x

def make_embedding(embedding_config):
    if embedding_config.function == 'constant':
        return Constant(embedding_config.config)
    elif embedding_config.function == 'fully_connected':
        return FullyConnectedEmbedding(embedding_config.config)
    else:
        raise ValueError("Unsupported embedding function! ({})".format(embedding_config.function))
