from collections import namedtuple
from .mpnn import make_mpnn, get_mpnn_config
from .flat import make_flat, get_flat_config

def get_model_generator(model_str):
    if model_str == 'mpnn':
        return make_mpnn
    elif model_str == 'flat':
        return make_flat

def get_model_config(model_str, args, dataset):
    if model_str == 'mpnn':
        return get_mpnn_config(args, dataset)
    elif model_str == 'flat':
        return get_flat_config(args, dataset)
