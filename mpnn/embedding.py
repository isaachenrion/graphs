import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, data_dim, state_dim):
        super().__init__()
        self.data_dim = data_dim
        self.state_dim = state_dim
        self.embedding = self.build_nn()

    def forward(self, x):
        return self.embedding(x)

    def build_nn(self):
        pass

class FullyConnectedEmbedding(Embedding):
    def __init__(self, *args):
        super().__init__(*args)

    def build_nn(self):
        return nn.Sequential(
            nn.Linear(self.data_dim, self.state_dim),
            nn.ReLU(),
            nn.Linear(self.state_dim, self.state_dim)
        )
