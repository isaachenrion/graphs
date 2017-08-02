from torch.autograd import Variable
import torch
def preprocess(data):
      examples = data['graphs']
      for i, G in enumerate(examples):
          for u in G.nodes():
              G.node[u]['data'] = Variable(torch.from_numpy(G.node[u]['data'])).unsqueeze(0).float()
          for u, v in G.edges():
              G.edge[u][v]['data'] = Variable(torch.from_numpy(G.edge[u][v]['data'])).unsqueeze(0).float()
          G.graph['readout'] = Variable(torch.from_numpy(G.graph['readout'])).unsqueeze(0).float()
      return examples
