import networkx as nx
import numpy as np
import pickle
import scipy.io
import os
from .path import DATA_DIR
from .datasets import GraphDataset, FixedOrderGraphDataset, Target
from .graph_utils import add_edge_data, add_graph_data, add_vertex_data, add_graph_data_dict

def generate_data(prefix, args):
    if prefix == 'train':
        num_examples = args.n_train
    else: num_examples = args.n_eval
    if args.problem == 'arithmetic':
        data = arithmetic(prefix, num_examples)
    elif args.problem == 'has_path':
        data = has_path(prefix, num_examples)
    elif args.problem == 'is_connected':
        data = is_connected(prefix, num_examples, args)
    elif args.problem == 'qm7_edge_representation':
        data = qm7(prefix, 'edge')
    elif args.problem == 'qm7_edge_representation_small':
        data = qm7_small(prefix, 'edge')
    elif args.problem == 'qm7_vertex_representation':
        data = qm7(prefix, 'vertex')
    elif args.problem == 'qm7_vertex_representation_small':
        data = qm7_small(prefix, 'vertex')
    elif args.problem == 'qm7b':
        data = qm7b(prefix)
    elif args.problem == 'qm7b_small':
        data = qm7b_small(prefix)
    elif args.problem == 'simple':
        data = simple(prefix, num_examples)
    else:
        raise ValueError("Problem was not recognised.")

    with open(os.path.join(DATA_DIR,'{}-{}.pkl'.format(args.problem, prefix)), 'wb') as f:
        pickle.dump(data, f)
    return None

def arithmetic(prefix='train', num_examples=1, debug=False):

    graph_targets = [
        Target('result', 'graph', 2)
    ]
    graphs = []
    vertex_dim = 5
    edge_dim = 3
    for i in range(num_examples):
        order = np.random.randint(5, 15)
        p = 0.5
        graph= nx.fast_gnp_random_graph(order, p)

        readout = 0.0
        for node in graph.nodes():
            node_data = np.reshape(np.random.randint(0, 2, vertex_dim), [1, -1])
            graph.add_node(node)
            graph.node[node]['data'] = node_data
            graph.add_edge(node, node, data=np.zeros([1, edge_dim]))

        for u, v in graph.edges():
            edge_matrix = np.expand_dims(2 * np.random.randint(0, 2, [vertex_dim, edge_dim]) - 1, 0)
            edge_data = np.matmul(graph.node[u]['data'] * graph.node[v]['data'], edge_matrix)
            graph.edge[u][v]['data'] = np.reshape(edge_data, [1, -1])
            graph.edge[u][v]['matrix'] = edge_matrix
            readout += edge_data.sum()

        graph.graph['result'] = np.expand_dims((readout > 0).astype('float32'), 1)

        graphs.append(graph)

    data = GraphDataset(
        graphs=graphs,
        problem_type="clf",
        vertex_dim=vertex_dim,
        edge_dim=edge_dim,
        graph_targets=graph_targets
    )
    return data

def has_path(prefix='train', num_examples=1, debug=False):

    graph_targets = [
        Target('has_path', 'graph', 1)
    ]
    graphs = []
    vertex_dim = 1
    edge_dim = 0
    readout_dim = 1
    for i in range(num_examples):
        order = np.random.randint(5, 15)
        graph = nx.fast_gnp_random_graph(order, 0.5)
        source = np.random.randint(0, order)
        target = np.random.randint(0, order)
        for node in graph.nodes():
            if node in [source, target]:
                graph.node[node]['data'] = np.ones([1, 1])
            else:
                graph.node[node]['data'] = np.zeros([1, 1])
        graph.graph['has_path'] = np.ones([1]) if nx.has_path(graph, source, target) else np.zeros([1])

        graphs.append(graph)
    data = GraphDataset(
        graphs=graphs,
        problem_type="clf",
        vertex_dim=vertex_dim,
        edge_dim=edge_dim,
        graph_targets=graph_targets,
    )
    return data

def simple(prefix='train', num_examples=1):
    graph_targets = [
        Target('result', 'graph', 2)
    ]
    graphs = []
    vertex_dim = 1
    edge_dim = 0
    mean_connected = 0.
    for i in range(num_examples):
        order = np.random.randint(5, 15)
        eps = 1e-5 * (2 * np.random.randint(0, 2) - 1)
        p = (1 + eps) * np.log(order) / order
        graph= nx.fast_gnp_random_graph(order, p)

        for node in graph.nodes():
            graph.node[node]['data'] = np.random.randint(0, 2, [1,1])
        graph.graph['result'] = np.ones([1]) if graph.node[0]['data'][0] == 0 else np.zeros([1])

        print(graph.graph['result'])
        graphs.append(graph)
    data = GraphDataset(
        graphs=graphs,
        problem_type="clf",
        vertex_dim=vertex_dim,
        edge_dim=edge_dim,
        graph_targets=graph_targets,
    )
    return data

def is_connected(prefix='train', num_examples=1, args=None):

    graph_targets = [
        Target('is_connected', 'graph', 2)
    ]
    graphs = []
    vertex_dim = 1
    edge_dim = 1
    mean_connected = 0.
    for i in range(num_examples):
        order = args.order if args is not None else 10

        eps = 1e-1 * (4 * np.random.randint(0, 2))
        p = (1 + eps) * np.log(order) / order
        G= nx.fast_gnp_random_graph(order, p)
        G.graph['is_connected'] = np.ones([1]) if nx.is_connected(G) else np.zeros([1])

        fgs = np.zeros([order, order])
        for u in G.nodes():
            G.node[u]['data'] = np.ones([1])
        for u, v in G.edges():
            G.edge[u][v]['data'] = np.ones([1])
            fgs[u][v] = 1
        G.graph['flat_graph_state'] = np.reshape(fgs, [-1])
        mean_connected += G.graph['is_connected']
        graphs.append(G)
    mean_connected /= num_examples
    print(mean_connected)
    data = FixedOrderGraphDataset(
        order=order,
        graphs=graphs,
        problem_type="clf",
        vertex_dim=vertex_dim,
        edge_dim=edge_dim,
        flat_graph_state_dim=G.graph['flat_graph_state'].shape[-1],
        graph_targets=graph_targets,
    )
    return data

def is_connected_padded(prefix='train', num_examples=1):
    data = is_connected(prefix, num_examples)
    data.pad_graphs()

def qm7(prefix, representation='edge', N=-1):
    n_atoms = 23
    if representation == 'vertex':
        vertex_dim = 4
        edge_dim = 0
    elif representation == 'edge':
        vertex_dim = 0
        edge_dim = 1
    graph_targets = [
        Target('E', 'graph', 1)
    ]

    data = scipy.io.loadmat(os.path.join(DATA_DIR, "qm7.mat"))
    T = np.expand_dims(data['T'][0], -1)

    # standardize
    edge_data = np.expand_dims(data['X'], -1)
    edge_data = edge_data - np.mean(edge_data, 0, keepdims=True)
    edge_data = edge_data / np.std(edge_data, 0, keepdims=True)

    vertex_data = np.concatenate([np.expand_dims(data['Z'],-1), data['R']], -1)
    vertex_data = vertex_data - np.mean(vertex_data, 0, keepdims=True)
    vertex_data = vertex_data / np.std(vertex_data, 0, keepdims=True)

    P = data['P']

    #import ipdb; ipdb.set_trace()

    graphs = []
    if prefix == 'train':
        good_indices = np.concatenate(P[1:], -1)[:N]
    elif prefix == 'eval':
        good_indices = P[0][:N]

    for i, idx in enumerate(good_indices):
        G = nx.complete_graph(n_atoms)
        if representation == 'vertex':
            add_vertex_data(G, vertex_data[idx])
            add_graph_data(G, np.reshape(vertex_data[idx], [-1]), key='flat_graph_state')

        elif representation == 'edge':
            add_edge_data(G, edge_data[idx])
            add_graph_data(G, np.reshape(edge_data[idx], [-1]), key='flat_graph_state')

        add_graph_data(G, T[idx], key='E')
        add_graph_data(G, vertex_dim, key='edge_dim')
        add_graph_data(G, edge_dim, key='vertex_dim')
        graphs.append(G)

    processed_data = FixedOrderGraphDataset(
        order=23,
        graphs=graphs,
        flat_graph_state_dim=G.graph['flat_graph_state'].shape[-1],
        problem_type="reg",
        vertex_dim=vertex_dim,
        edge_dim=edge_dim,
        graph_targets=graph_targets
    )
    return processed_data

def qm7_small(prefix, representation='edge'):
    return qm7(prefix, representation,100)

def qm7b(prefix, N=-1):
    n_atoms = 23

    data = scipy.io.loadmat(os.path.join(DATA_DIR, "qm7b.mat"), chars_as_strings=True)
    edge_data = np.expand_dims(data['X'], -1)

    # standardize
    edge_data = edge_data - np.mean(edge_data, 0, keepdims=True)
    edge_data = edge_data / np.std(edge_data, 0, keepdims=True)

    T = data['T']
    #T_mean = np.mean(T, 0, keepdims=True)
    #T_std = np.std(T, 0, keepdims=True)
    #T = (T - T_mean) / T_std


    graph_targets = [
        Target('E-PBE0', 'graph', 1),
        Target('E-max-ZINDO', 'graph', 1),
        Target('I-max-ZINDO', 'graph', 1),
        Target('HOMO-ZINDO', 'graph', 1),
        Target('LUMO-ZINDO', 'graph', 1),
        Target('E-1st-ZINDO', 'graph', 1),
        Target('IP-ZINDO', 'graph', 1),
        Target('EA-ZINDO', 'graph', 1),
        Target('HOMO-PBE0', 'graph', 1),
        Target('LUMO-PBE0', 'graph', 1),
        Target('HOMO-GW', 'graph', 1),
        Target('LUMO-GW', 'graph', 1),
        Target('alpha-PBE0', 'graph', 1),
        Target('alpha-SCS', 'graph', 1),
    ]


    # make validation set
    q7data = scipy.io.loadmat(os.path.join(DATA_DIR, "qm7.mat"))
    P = q7data['P']
    if prefix == 'train':
        good_indices = np.concatenate(P[1:], -1)[:N]
    elif prefix == 'eval':
        good_indices = P[0][:N]

    graphs = []
    for i, idx in enumerate(good_indices):
        G = nx.Graph()
        add_edge_data(G, edge_data[idx])
        #add_graph_data(G, T[idx], key='readout')
        add_graph_data(G, np.reshape(edge_data[idx], [-1]), key='flat_graph_state')
        graph_data_dict = {target.name: T[idx][j] for j, target in enumerate(graph_targets)}
        add_graph_data_dict(G, graph_data_dict)

        graphs.append(G)



    processed_data = FixedOrderGraphDataset(
        order=n_atoms,
        graphs=graphs,
        flat_graph_state_dim=G.graph['flat_graph_state'].shape[-1],
        problem_type="reg",
        vertex_dim=0,
        edge_dim= 1,
        graph_targets=graph_targets,

    )
    return processed_data

def qm7b_small(prefix):
    return qm7b(prefix, 100)
