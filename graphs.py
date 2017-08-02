import networkx as nx
import numpy as np
import pickle

def generate(prefix='train', num_examples=1, graph_type=None, debug=False):

    if graph_type == 'petersen':
        graph_class = nx.petersen_graph
    elif graph_type == 'cubical':
        graph_class = nx.cubical_graph
    elif graph_type == 'random':
        def graph_class():
            i = np.random.randint(5, 15)
            return nx.fast_gnp_random_graph(i, 0.5)

    graphs = []
    node_dim = 5
    edge_dim = 3
    readout_dim = 1
    for i in range(num_examples):
        graph=graph_class()
        #print(graph)

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


        graph.graph['readout'] = np.expand_dims(readout > 0, 1).astype('float32')

        # sanity check
        readout = 0.0
        for u, v in graph.edges():
            d_u = graph.node[u]['data']
            d_v = graph.node[v]['data']
            edge_matrix = graph.edge[u][v]['matrix']
            edge_data = graph.edge[u][v]['data']
            readout += edge_data.sum()
            assert np.all(edge_data == np.dot(edge_matrix, d_u * d_v))
        assert np.expand_dims(readout > 0, 1).astype('float32') == graph.graph['readout']

        graphs.append(graph)
    data = {"vertex_dim": node_dim, "edge_dim": edge_dim, "readout_dim":readout_dim, "graphs":graphs}

    if debug:
        import matplotlib.pyplot as plt
        nx.draw(graph)
        plt.show()

    with open('data/{}-{}.pkl'.format(graph_type, prefix), 'wb') as f:
        pickle.dump(data, f)
