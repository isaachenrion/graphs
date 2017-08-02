
"""
train.py

Core training script for the addition task-specific NPI. Instantiates a model, then trains using
the precomputed data.
"""
import pickle
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from config import DEFAULTCONFIG
import numpy as np
import time
import datetime
import itertools
import os
from mpnn import MPNN
from utils import preprocess
from evaluate import evaluate

DATA_DIR = "data"
EXP_DIR = "experiments"

def train(epochs, graph_type, lr):
    # get timestamp for model id
    dt = datetime.datetime.now()
    timestamp = '{}-{}___{:02d}-{:02d}-{:02d}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second)
    os.makedirs(os.path.join(EXP_DIR, timestamp))

    # load train data
    train_data_path = os.path.join(DATA_DIR, graph_type + '-train.pkl')
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    train_examples = preprocess(train_data)

    # load eval data
    eval_data_path = os.path.join(DATA_DIR, graph_type + '-eval.pkl')
    with open(eval_data_path, 'rb') as f:
        eval_data = pickle.load(f)
    eval_examples = preprocess(eval_data)


    # Initialize MPNN Model
    print('Initializing MPNN Model!')
    vertex_dim = train_data['vertex_dim']
    edge_dim = train_data['edge_dim']
    readout_dim = train_data['readout_dim']
    state_dim = 11
    message_dim = 13
    n_iters = 3
    model = MPNN(vertex_dim, state_dim, edge_dim, message_dim, readout_dim, n_iters, DEFAULTCONFIG)
    print(model)

    # Loss functions
    loss_fn = nn.BCELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # global step
    global_step = 0

    # Start Training
    for ep in range(1, epochs + 1):
        train_results = train_one_epoch(model, train_examples, loss_fn, optimizer)

        out_str = "\n"
        out_str += "Epoch {:02d} took {:.1f} seconds\n".format(ep, train_results['t'])
        out_str += "Training loss: {:.5f}, Training accuracy: {:.2f}".format(train_results['loss'], train_results['acc'])
        print(out_str)

        global_step += len(train_examples)
        if global_step % 10000 == 0:
            lr *= 0.95
            optimizer = optim.Adam(model.parameters(), lr=lr)

        if ep % 5 == 0:
            eval_results = evaluate(model, eval_examples, loss_fn)
            out_str = "\n"
            out_str += "Evaluation on {:02d} examples took {:.1f} seconds\n".format(len(eval_examples), eval_results['t'])
            out_str += "Evaluation loss: {:.5f}, Evaluation accuracy: {:.2f}".format(eval_results['loss'], eval_results['acc'])
            print(out_str)

            # Save Model
            torch.save(model, 'experiments/{}/model.ckpt'.format(timestamp))
    return model

def train_one_epoch(model, examples, loss_fn, optimizer):
    epoch_loss = 0.0
    epoch_acc = 0.0
    t0 = time.time()

    for i, G in enumerate(examples):
        # reset hidden states
        for u in G.nodes():
            G.node[u]['hidden'] = Variable(torch.zeros(1, model.state_dim))
            G.node[u]['message'] = Variable(torch.zeros(1, model.message_dim))

        # clear buffers
        optimizer.zero_grad()

        # forward model
        readout = model(G)

        # get loss
        loss = loss_fn(readout, G.graph['readout'])

        # stats
        hard_prediction = readout.data.numpy()[0,0] > 0.5
        ground_truth = G.graph['readout'].data.numpy()[0,0]
        epoch_acc += (hard_prediction == ground_truth)

        epoch_loss += loss.data.numpy()[0]

        # backward losses from trajectory
        loss.backward()

        # optimization
        optimizer.step()

    t = time.time() - t0
    return {"loss": epoch_loss / len(examples), "acc":epoch_acc/len(examples), "t": t}

# wrap
def wrap(x, unsqueeze=True, t_type='float'):
    x_ = Variable(torch.from_numpy(x))

    if t_type == 'float':
        x_ = x_.float()
    elif t_type == 'long':
        x_ = x_.long()
    else:
        raise ValueError("Only float and long are supported atm.")

    if unsqueeze:
        x_ = x_.unsqueeze(0)
    return x_

def unwrap(x):
    out = x.data.numpy()
    if np.prod(out.shape) == 1:
        return out.flatten()[0]
    return out
