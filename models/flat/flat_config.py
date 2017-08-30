from collections import namedtuple

FlatConfig = namedtuple(
        'FlatConfig', [
            'state_dim',
            'hidden_dim',
            'graph_targets',
            'mode'
        ]
)

def get_flat_config(args, dataset):
    config = FlatConfig(
        state_dim=dataset.flat_graph_state_dim,
        hidden_dim=args.hidden_dim,
        graph_targets=dataset.graph_targets,
        mode=dataset.problem_type
    )
    return config
