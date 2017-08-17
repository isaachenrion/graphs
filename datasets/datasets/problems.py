import networkx as nx
import numpy as np
import pickle
import scipy.io
import os

DATA_DIR = 'data'
def generate(prefix, problem, num_examples):
    if problem == 'arithmetic':
        data = arithmetic(prefix, num_examples)
    elif problem == 'has_path':
        data = has_path(prefix, num_examples)
    elif problem == 'is_connected':
        data = is_connected(prefix, num_examples)
    elif problem == 'qm7':
        data = qm7(prefix)
    elif problem == 'qm7_small':
        data = qm7_small(prefix)
    else:
        raise ValueError("Problem was not recognised.")

    with open('data/{}-{}.pkl'.format(problem, prefix), 'wb') as f:
        pickle.dump(data, f)
    return None

def arithmetic(prefix='train', num_examples=1, debug=False):

    graphs = []
    node_dim = 5
    edge_dim = 3
    readout_dim = 1
    for i in range(num_examples):
        order = np.random.randint(5, 15)
        p = 0.5
        graph= nx.fast_gnp_random_graph(order, p)

        readout = 0.0
        for node in graph.nodes():
            node_data = np.random.randint(0, 2, node_dim)
            graph.add_node(node)
            graph.node[node]['data'] = node_data

        for u, v in graph.edges():
            edge_matrix = 2 * np.random.randint(0, 2, [edge_dim, node_dim]) - 1
            edge_data = np.dot(edge_matrix, graph.node[u]['data'] * graph.node[v]['data'])
            graph.edge[u][v]['data'] = edge_data
            graph.edge[u][v]['matrix'] = edge_matrix
            readout += edge_data.sum()

        graph.graph['readout'] = np.expand_dims(readout, 1)

        graphs.append(graph)
    data = {"vertex_dim": node_dim, "edge_dim": edge_dim, "readout_dim":readout_dim, "graphs":graphs, "problem_type":"reg"}
    return data

def has_path(prefix='train', num_examples=1, debug=False):

    graphs = []
    node_dim = 1
    edge_dim = 1
    readout_dim = 1
    for i in range(num_examples):
        order = np.random.randint(5, 15)
        graph = nx.fast_gnp_random_graph(order, 0.5)

        source = np.random.randint(0, order)
        target = np.random.randint(0, order)
        for node in graph.nodes():
            if node in [source, target]:
                graph.node[node]['data'] = np.ones([1])
            else:
                graph.node[node]['data'] = np.zeros([1])

        for u, v in graph.edges():
            graph.edge[u][v]['data'] = np.zeros([1])

        graph.graph['readout'] = np.ones([1]) if nx.has_path(graph, source, target) else np.zeros([1])

        graphs.append(graph)
    data = {"vertex_dim": node_dim, "edge_dim": edge_dim, "readout_dim":readout_dim, "graphs":graphs, "problem_type":"clf"}
    return data

def is_connected(prefix='train', num_examples=1, debug=False):

    graphs = []
    node_dim = 1
    edge_dim = 1
    readout_dim = 1
    mean_connected = 0.
    for i in range(num_examples):
        order = np.random.randint(5, 15)
        eps = 1e-5 * (2 * np.random.randint(0, 2) - 1)
        p = (1 + eps) * np.log(order) / order
        graph= nx.fast_gnp_random_graph(order, p)


        for node in graph.nodes():
            graph.node[node]['data'] = np.ones([1])

        for u, v in graph.edges():
            graph.edge[u][v]['data'] = np.zeros([1])

        graph.graph['readout'] = np.ones([1]) if nx.is_connected(graph) else np.zeros([1])
        mean_connected += graph.graph['readout']
        graphs.append(graph)
    mean_connected /= num_examples
    print(mean_connected)
    data = {"vertex_dim": node_dim, "edge_dim": edge_dim, "readout_dim":readout_dim, "graphs":graphs, "problem_type":"clf"}
    return data

def qm7(prefix, N=-1):
    n_atoms = 23
    data = scipy.io.loadmat(os.path.join(DATA_DIR, "qm7.mat"))
    edge_data = np.expand_dims(data['X'], -1)

    # standardize
    edge_data = edge_data - np.mean(edge_data, 0, keepdims=True)
    edge_data = edge_data / np.std(edge_data, 0, keepdims=True)
    T = np.expand_dims(data['T'][0], -1)

    Z = data['Z']
    R = data['R']

    P = data['P']

    graphs = []
    if prefix == 'train':
        good_indices = np.concatenate(P[1:], -1)[:N]
    elif prefix == 'eval':
        good_indices = P[0][:N]

    for i, idx in enumerate(good_indices):
        G = nx.Graph()
        add_edge_data(G, edge_data[idx])
        add_graph_data(G, T[idx], key='readout')
        add_graph_data(G, np.reshape(edge_data[idx], [-1]), key='flat_graph_state')
        graphs.append(G)

    processed_data = FixedOrderGraphDataset(
        order=23,
        graphs=graphs,
        flat_graph_state_dim=G.graph['flat_graph_state'].shape[-1],
        problem_type="reg",
        vertex_dim=0,
        edge_dim= 1,
        readout_dim=1,
    )
    return processed_data

def qm7_small(prefix):
    return qm7(prefix, 100)
