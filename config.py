
from mpnn import *
from collections import namedtuple

Config = namedtuple('Config', ['message_function', 'vertex_update_function', 'readout_function', 'embedding_function'])

DEFAULTCONFIG = Config(
                message_function=FullyConnectedMessage,
                vertex_update_function=GRUVertexUpdate,
                readout_function=FullyConnectedReadout,
                embedding_function=FullyConnectedEmbedding
                )
