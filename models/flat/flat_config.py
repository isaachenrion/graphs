from collections import namedtuple

FlatConfig = namedtuple(
        'FlatConfig', [
            'state_dim',
            'hidden_dim',
            'readout_dim',
            'mode'
        ]
)

def get_flat_config(args, dataset):
    config = FlatConfig(
        state_dim=dataset.flat_graph_state_dim,
        hidden_dim=500,
        readout_dim=dataset.readout_dim,
        mode=dataset.problem_type
    )
    return config
