import os
import pickle
from .datasets import  BatchedFixedOrderGraphDataset, FixedOrderGraphDataset, GraphDataset, BatchedGraphDataset
from .add_virtual_node import add_virtual_node, add_target_nodes
from .path import DATA_DIR

def load_from_path(data_path, args):
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    if isinstance(dataset, FixedOrderGraphDataset):
        dataset = BatchedFixedOrderGraphDataset(dataset, args.batch_size)
    elif isinstance(dataset, GraphDataset):
        dataset = BatchedGraphDataset(dataset, args.batch_size)
    if args.model == 'vcn':
        add_target_nodes(dataset)


    dataset = dataset.preprocess()
    return dataset

def load_data(args):
    train_data_path = os.path.join(DATA_DIR, args.problem + '-train.pkl')
    eval_data_path = os.path.join(DATA_DIR, args.problem + '-eval.pkl')

    training_set = load_from_path(train_data_path, args)
    validation_set = load_from_path(eval_data_path, args)

    return training_set, validation_set
