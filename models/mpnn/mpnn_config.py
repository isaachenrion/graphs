
from .embedding import make_embedding
from .message import *
from .readout import *
from .vertex_update import *
from collections import namedtuple



FunctionAndConfig = namedtuple(
        'FunctionAndConfig', [
            'function',
            'config'
        ]
)

MPNNConfig = namedtuple(
        'MPNNConfig', [
            'message',
            'vertex_update',
            'readout',
            'embedding',
            'n_iters',
            'parallelism'
        ]
)

MessageConfig = namedtuple(
        'MessageConfig', [
            'hidden_dim',
            'edge_dim',
            'message_dim',
        ]
)

VertexUpdateConfig = namedtuple(
        'VertexUpdateConfig', [
            'vertex_state_dim',
            'edge_dim',
            'hidden_dim',
            'message_dim'
        ]
)

EmbeddingConfig = namedtuple(
        'EmbeddingConfig', [
            'data_dim',
            'state_dim'
        ]
)

ReadoutConfig = namedtuple(
        'ReadoutConfig', [
            'hidden_dim',
            'readout_hidden_dim',
            'mode',
            'graph_targets',
        ]
)

def get_mpnn_config(args, dataset):
    if args.message == 'constant':
        args.message_dim = args.hidden_dim + dataset.edge_dim
    config = MPNNConfig(
        message=FunctionAndConfig(
            function=args.message,
            config=MessageConfig(
                hidden_dim=args.hidden_dim,
                edge_dim=dataset.edge_dim,
                message_dim=args.message_dim,
            )
        ),
        vertex_update=FunctionAndConfig(
            function=args.vertex_update,
            config=VertexUpdateConfig(
                vertex_state_dim=args.vertex_state_dim if args.vertex_state_dim != 0 else dataset.vertex_dim,
                edge_dim=dataset.edge_dim,
                hidden_dim=args.hidden_dim,
                message_dim=args.message_dim
            )
        ),
        readout=FunctionAndConfig(
            function=args.readout,
            config=ReadoutConfig(
                hidden_dim=args.hidden_dim,
                readout_hidden_dim=10,
                mode=dataset.problem_type,
                graph_targets=dataset.graph_targets,
            )
        ),
        embedding=FunctionAndConfig(
            function=args.embedding,
            config=EmbeddingConfig(
                data_dim=dataset.vertex_dim,
                state_dim=args.vertex_state_dim
            )
        ),
        n_iters=args.n_iters,
        parallelism=args.parallelism
    )
    return config
