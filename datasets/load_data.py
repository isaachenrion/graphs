import os
import pickle
from .datasets import preprocess, BatchedFixedOrderGraphDataset
from .path import DATA_DIR

def load_data(problem):
    train_data_path = os.path.join(DATA_DIR, problem + '-train.pkl')
    with open(train_data_path, 'rb') as f:
        training_set = pickle.load(f)
    training_set = BatchedFixedOrderGraphDataset(training_set, 13)
    training_set = preprocess(training_set)

    # load eval data
    eval_data_path = os.path.join(DATA_DIR, problem + '-eval.pkl')
    with open(eval_data_path, 'rb') as f:
        validation_set = pickle.load(f)
    validation_set = BatchedFixedOrderGraphDataset(validation_set, 13)
    validation_set = preprocess(validation_set)
    return training_set, validation_set
