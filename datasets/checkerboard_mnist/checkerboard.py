import numpy as np
import networkx as nx

def checkerboard_dataset(images):
    graphs = []
    for image in images:
        G = grid_graph(image.shape[0], image.shape[1], image)
        graphs.append(G)



def grid_graph(length, width, vertex_data=None):
    G = nx.Graph()
    for i in range(length):
        for j in range(width):
            G.add_node((i, j))
            if vertex_data is not None:
                G.node[(i, j)] = vertex_data[i, j]

    for i in range(length - 1):
        for j in range(width - 1):
            G.add_edge((i, j), (i + 1, j))
            G.add_edge((i, j), (i, j + 1))

    return G
