from torch.autograd import Variable
import torch
from torch.utils.data import Dataset
import copy
import networkx as nx
import numpy as np

def preprocess(graph_dataset):
    wrapped_dataset = copy.deepcopy(graph_dataset)
    for i, G in enumerate(wrapped_dataset):
        if graph_dataset.has_vertex_data():
            for u in G.nodes():
                G.node[u]['data'] = Variable(torch.from_numpy(G.node[u]['data'])).float()
        if graph_dataset.has_edge_data():
            for u, v in G.edges():
                G.edge[u][v]['data'] = Variable(torch.from_numpy(G.edge[u][v]['data'])).float()
        if graph_dataset.has_graph_data():
            G.graph['readout'] = Variable(torch.from_numpy(G.graph['readout'])).float()
            G.graph['flat_graph_state'] = Variable(torch.from_numpy(G.graph['flat_graph_state'])).float()


    return wrapped_dataset


class GraphDataset(Dataset):
    def __init__(self, graphs=None, problem_type=None, vertex_dim=None, edge_dim=None, readout_dim=None):
        super().__init__()
        self.graphs = graphs
        self.problem_type = problem_type
        self.vertex_dim = vertex_dim
        self.edge_dim = edge_dim
        self.readout_dim = readout_dim

    def set_graphs(self, graphs):
        assert len(graphs) == len(self)
        self.graphs = graphs

    def has_vertex_data(self):
        return self.vertex_dim > 0

    def has_edge_data(self):
        return self.edge_dim > 0

    def has_graph_data(self):
        return self.readout_dim > 0

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]

class FixedOrderGraphDataset(GraphDataset):
    def __init__(self, order=None, flat_graph_state_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.order = order
        self.flat_graph_state_dim = flat_graph_state_dim

class BatchedFixedOrderGraphDataset(FixedOrderGraphDataset):
    def __init__(self, graph_dataset, batch_size):
        super().__init__(
            graph_dataset.order,
            graph_dataset.graphs,
            graph_dataset.problem_type,
            graph_dataset.flat_graph_state_dim,
            graph_dataset.vertex_dim,
            graph_dataset.edge_dim,
            graph_dataset.readout_dim,
        )
        self.model_graph = nx.create_empty_copy(graph_dataset.graphs[0])
        self.make_batches(batch_size)

    def make_batches(self, batch_size):

        self.batch_size = batch_size
        self.n_batches, self.remainder = divmod(len(self.graphs), batch_size)

        def create_empty_batch_graph(batch_size):
            empty_batch_graph = nx.create_empty_copy(self.model_graph)
            if self.vertex_dim > 0: add_vertex_data(empty_batch_graph, np.zeros([self.order, batch_size, self.vertex_dim]))
            if self.edge_dim > 0: add_edge_data(empty_batch_graph, np.zeros([self.order, self.order, batch_size, self.edge_dim]))
            if self.readout_dim > 0: add_graph_data(empty_batch_graph, np.zeros([batch_size, self.readout_dim]), 'readout')
            add_graph_data(empty_batch_graph, np.zeros([batch_size, self.flat_graph_state_dim]), 'flat_graph_state')
            return empty_batch_graph

        self.batches = [create_empty_batch_graph(self.batch_size) for _ in range(self.n_batches)]

        on_batch = 0

        for i, G in enumerate(self.graphs):
            batch_idx = i % self.batch_size
            if self.vertex_dim > 0:
                for u in G.nodes():
                    self.batches[on_batch].node[u]['data'][batch_idx] = G.node[u]['data']

            if self.edge_dim > 0:
                for u, v in G.edges():
                    self.batches[on_batch].edge[u][v]['data'][batch_idx] = G.edge[u][v]['data']

            if self.readout_dim > 0:
                self.batches[on_batch].graph['readout'][batch_idx] = G.graph['readout']

            self.batches[on_batch].graph['flat_graph_state'][batch_idx] = G.graph['flat_graph_state']
            if batch_idx + 1 == batch_size:
                on_batch += 1
            if batch_size * on_batch + self.remainder == len(self.graphs):
                break



    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[index]
