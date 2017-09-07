
import numpy as np
import networkx as nx
import copy

def add_edge_data(G, edge_data_matrix, key='data'):
    for i, _ in enumerate(edge_data_matrix):
        for j, _ in enumerate(edge_data_matrix):
            data = edge_data_matrix[i, j]
            G.add_edge(i, j)
            G.edge[i][j][key] = data

def add_vertex_data(G, vertex_data, key='data'):
    for i, data in enumerate(vertex_data):
        G.add_node(i)
        G.node[i][key] = data

def add_vertex_data_dict(G, vertex_data_dict):
    for k, v in vertex_data_dict.items():
        G.add_node(k)
        G.node[k]['data'] = v

def add_graph_data(G, graph_data, key='data'):
    G.graph[key] = graph_data

def add_graph_data_dict(G, graph_data_dict):
    for k, v in graph_data_dict.items():
        add_graph_data(G, v, k)

def fully_connected_padding(G):
    '''Given a graph G with edge data, make it fully connected
    with zeros on the new edges, and an extra bit to distinguish
    new edges from old.
    '''
    G_ = copy.deepcopy(G)

    for u, v in G.edges():
        if G.graph['edge_dim'] != 0:
            temp = G.edge[u][v]['data']
            G_.edge[u][v]['data'] = np.concatenate((temp, np.zeros(G.graph['batch_size'], 1)), 1)
        else:
            G_.edge[u][v]['data'] = np.zeros(G.graph['batch_size'], 1)

    for u, v in nx.complement(G).edges():
        G_.add_edge(u, v)
        if G.graph['edge_dim'] != 0:
            temp = np.zeros(G.graph['batch_size'], G.graph['edge_dim'] + 1)
            temp[:, -1] = 1
            G_.edge[u][v]['data'] = temp
        else:
            G_.edge[u][v]['data'] = np.ones(G.graph['batch_size'], 1)

    G_.graph['edge_dim'] += 1

    return G_
