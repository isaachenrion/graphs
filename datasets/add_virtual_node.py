import numpy as np
def add_virtual_node(batched_fixed_order_graph_dataset):
    ds = batched_fixed_order_graph_dataset
    for i, G in enumerate(ds):
        G.add_node('virtual')
        if ds.vertex_dim > 0:
              G.node['virtual']['data'] = np.zeros([G.graph['batch_size'], ds.vertex_dim])
        for u in range(G.order):
            G.add_edge('virtual', u)
            if ds.edge_dim > 0:
                G.edge['virtual'][u]['data'] = np.zeros([G.graph['batch_size'], ds.edge_dim])
    return None

def add_target_nodes(graph_dataset):
    targets = graph_dataset.graph_targets
    for i, G in enumerate(graph_dataset):
        try:
            bs = G.graph['batch_size']
        except KeyError:
            bs = 1

        order = len(G.nodes())
        for target in targets:
            G.add_node(target.name)
            if graph_dataset.vertex_dim > 0:
                  G.node[target.name]['data'] = np.zeros([bs, graph_dataset.vertex_dim])
        for target in targets:
            for u in G.nodes():
                G.add_edge(target.name, u)
                if graph_dataset.edge_dim > 0:
                    G.edge[target.name][u]['data'] = np.zeros([bs, graph_dataset.edge_dim])
    return None
