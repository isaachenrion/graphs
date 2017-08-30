
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
