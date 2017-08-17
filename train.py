
"""
train.py

Core training script for the addition task-specific NPI. Instantiates a model, then trains using
the precomputed data.
"""
import pickle
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from experiment_config import get_experiment_config
import numpy as np
import time
import datetime
import itertools
import copy
import os
from models import *
from datasets import load_data
from evaluate import evaluate
import networkx as nx

EXP_DIR = "experiments"

def train(args):
    # get timestamp for model id
    dt = datetime.datetime.now()
    timestamp = '{}-{}___{:02d}-{:02d}-{:02d}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second)
    model_dir = os.path.join(EXP_DIR, timestamp)
    os.makedirs(model_dir)

    if args.verbosity == 0:
        def _print(x):
            pass
    elif args.verbosity == 1:
        def _print(x):
            print(x)
    elif args.verbosity == 2:
        outfile = open(os.path.join(model_dir, 'log.txt'), 'w')
        def _print(x):
            print(x)
            outfile.write(str(x))

    for k, v in vars(args).items(): print('{} : {}'.format(k, v))

    # load train data
    _print('Loading data...')
    training_set, validation_set = load_data(args.problem)

    experiment_config = get_experiment_config(args, training_set)

    # Initialize MPNN Model
    if args.load is None:

        _print('Initializing model...')

        model_generator = experiment_config.model_generator
        model_config = experiment_config.model_config
        model = model_generator(model_config)

        #if args.model == 'mpnn':
        #    model = make_mpnn(config)
        #elif args.model == 'flat':
        #    model = Flat(
        #        state_dim=training_set.flat_graph_state_dim,
        #        hidden_dim=500,
        #        readout_dim=training_set.readout_dim,
        #        mode=mode
        #    )

    else:
        _print('Loading model from {}!'.format(args.load))
        model = torch.load(os.path.join(EXP_DIR, args.load, 'model.ckpt'))
    _print(model)

    # Loss functions
    #if mode == 'clf': # classification
    #    loss_fn = nn.BCELoss()
    #    monitors = CLF_MONITORS
    #elif mode == 'reg': # regression
    #    loss_fn = nn.MSELoss()
    #    monitors = REG_MONITORS
    loss_fn = experiment_config.loss_fn
    monitors = experiment_config.monitors
    _print(loss_fn)

    # Optimizer
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.5, patience=3)
    _print(optimizer)

    # global step
    global_step = 0

    # Start Training
    for epoch in range(1, args.epochs + 1):
        results = train_one_epoch(model, training_set, loss_fn, optimizer, monitors)
        out_str = results_str(epoch, results, 'train')
        _print(out_str)

        if epoch % 5 == 0:
            results = evaluate(model, validation_set, loss_fn, monitors)
            scheduler.step(results['loss'])
            out_str = results_str(epoch, results, 'eval')
            _print(out_str)
            torch.save(model, os.path.join(model_dir, 'model.ckpt'))
            _print("Saved model to {}".format(os.path.join(model_dir, 'model.ckpt')))
    return model

def results_str(epoch, results, run_mode):
    out_str = ""
    if run_mode == 'train':
        out_str += "Epoch {}\n".format(epoch)
    for k, v in results.items():
        out_str += "{} {}: {:.5f}\n".format(run_mode, k, v)
    if run_mode == 'eval':
        pad_str = '\n{s:{c}^{n}}\n'.format(s='#',n=20,c='#')
        out_str += pad_str
    return out_str

def train_one_epoch(model, dataset, loss_fn, optimizer, monitors):
    batch_size = 1
    epoch_loss = 0.0
    epoch_acc = 0.0
    t0 = time.time()
    epoch_stats = {name: 0.0 for name in monitors.keys()}

    for i, G_batch in enumerate(dataset):

        # reset hidden states
        G_batch = model.reset_hidden_states(G_batch)

        # clear buffers
        optimizer.zero_grad()

        # forward model
        readout = model(G_batch)

        # get loss, backward and optimize
        loss = loss_fn(readout, G_batch.graph['readout'])
        loss.backward()
        optimizer.step()

        # stats
        stats = {name: monitor(G_batch, readout) for name, monitor in monitors.items()}
        epoch_stats = {name: (epoch_stats[name] + stats[name]) for name in monitors.keys()}

        if i % 10 == 0 and False:
            #G_ = copy.copy(G)
            #for u, v in G_.edges():
            #    data = G_.edge[u][v]['data']
            #    G_.edge[u][v]['data'] = np.round(data.data.numpy()[0], 2).item()
            #am = nx.adjacency_matrix(G, weight='data')
            #print(am.todense())
            print("TARGET = {}".format(G_batch.graph['readout'].data.numpy()[0]))
            print("READOUT = {}".format(readout.data.numpy()[0]))
            print("")


    t = time.time() - t0
    epoch_stats = {name: stat / len(dataset) for name, stat in epoch_stats.items()}
    epoch_stats["time"] = t
    return epoch_stats
