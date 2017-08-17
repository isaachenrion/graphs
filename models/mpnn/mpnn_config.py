
from .embedding import *
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
        ]
)

MessageConfig = namedtuple(
        'MessageConfig', [
            'hidden_dim',
            'edge_dim',
            'message_dim'
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
            'readout_dim',
            'mode'
        ]
)

def get_mpnn_config(args, dataset):

    config = MPNNConfig(
        message=FunctionAndConfig(
            function=FullyConnectedMessage,
            config=MessageConfig(
                hidden_dim=args.hidden_dim,
                edge_dim=dataset.edge_dim,
                message_dim=args.message_dim
            )
        ),
        vertex_update=FunctionAndConfig(
            function=GRUVertexUpdate,
            config=VertexUpdateConfig(
                vertex_state_dim=args.vertex_state_dim,
                edge_dim=dataset.edge_dim,
                hidden_dim=args.hidden_dim,
                message_dim=args.message_dim
            )
        ),
        readout=FunctionAndConfig(
            function=FullyConnectedReadout,
            config=ReadoutConfig(
                hidden_dim=args.hidden_dim,
                readout_hidden_dim=10,
                readout_dim=dataset.readout_dim,
                mode=dataset.problem_type
            )
        ),
        embedding=FunctionAndConfig(
            function=FullyConnectedEmbedding,
            config=EmbeddingConfig(
                data_dim=dataset.vertex_dim,
                state_dim=args.vertex_state_dim
            )
        ),
        n_iters=args.n_iters,
    )
    return config
