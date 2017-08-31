
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
import networkx as nx
import torch.nn.functional as F

EXP_DIR = "experiments"

def train(args):
    global DEBUG
    DEBUG = args.debug

    # get timestamp for model id
    dt = datetime.datetime.now()
    timestamp = '{}-{}___{:02d}-{:02d}-{:02d}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second)
    model_dir = os.path.join(EXP_DIR, timestamp)
    os.makedirs(model_dir)

    _print = get_print_function(model_dir, args.verbosity)

    # set device (if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.device(args.gpu)

    # write the args to outfile
    for k, v in vars(args).items(): _print('{} : {}\n'.format(k, v))

    # load data
    training_set, validation_set = load_data(args)
    _print('Loaded data: {} training examples, {} validation examples\n'.format(
        len(training_set.graphs), len(validation_set.graphs)))

    # get config
    experiment_config = get_experiment_config(args, training_set)

    # initialize model
    if args.load is None:
        _print('Initializing model...\n')
        model = experiment_config.model_generator(experiment_config.model_config)
    else:
        _print('Loading model from {}\n'.format(args.load))
        model = torch.load(os.path.join(EXP_DIR, args.load, 'model.ckpt'))
    if torch.cuda.is_available():
        model.cuda()
    _print(model)
    _print('Training loss: {}\n'.format(experiment_config.loss_fn))

    # optimizer
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    #optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.5, patience=3)
    _print(optimizer)
    _print(scheduler)

    # Start Training
    for epoch in range(1, args.epochs + 1):
        results = train_one_epoch(model, training_set, experiment_config.loss_fn, optimizer, experiment_config.monitors, args.debug)
        _print(results_str(epoch, results, 'train'))

        if epoch % 5 == 0:
            results = evaluate_one_epoch(model, validation_set, experiment_config.loss_fn, experiment_config.monitors)
            _print(results_str(epoch, results, 'eval'))

            torch.save(model, os.path.join(model_dir, 'model.ckpt'))
            _print("Saved model to {}\n".format(os.path.join(model_dir, 'model.ckpt')))

            scheduler.step(results['loss'])
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

def unwrap(variable_dict):
    return {name: var.data.cpu().numpy().item() for name, var in variable_dict.items()}

def train_one_batch_serial(model, batch, loss_fn, optimizer, monitors):
    batch_loss = Variable(torch.zeros(1))
    batch_stats = {name: 0.0 for name in monitors.names}
    optimizer.zero_grad()
    for i, G in enumerate(batch):
        # reset hidden states
        model.reset_hidden_states(G)
        # forward model
        model_output = model(G)
        # get loss
        loss = loss_fn(model_output, G)
        batch_loss += loss
        # get stats
        stats = unwrap(monitors(model_output, G))
        batch_stats = {name: (batch_stats[name] + stats[name]) for name in monitors.names}

    batch_loss = batch_loss / len(batch)
    batch_loss.backward()

    torch.nn.utils.clip_grad_norm(model.parameters(), .1)
    optimizer.step()

    batch_stats = {name: batch_stats[name] / len(batch) for name in monitors.names}
    return batch_stats

def train_one_batch_parallel(model, batch, loss_fn, optimizer, monitors):

    optimizer.zero_grad()
    model.reset_hidden_states(batch)

    # forward model
    model_output = model(batch)

    # get loss
    batch_loss = loss_fn(model_output, batch)

    #import ipdb; ipdb.set_trace()

    # backward and optimize
    batch_loss.backward()

    if False and DEBUG:
        for k, v in model_output.items():
            print("Output variance: {}".format(v.var(0).data.numpy()[0]))
            print("Target variance: {}".format(batch.graph[k].var(0).data.numpy()[0]))

    torch.nn.utils.clip_grad_norm(model.parameters(), .1)
    optimizer.step()

    # get stats
    batch_stats = unwrap(monitors(model_output, batch))

    return batch_stats

def train_one_epoch(model, dataset, loss_fn, optimizer, monitors, debug):
    t0 = time.time()
    epoch_stats = {name: 0.0 for name in monitors.names}

    if dataset.order is None:
        train_one_batch = train_one_batch_serial
    else:
        train_one_batch = train_one_batch_parallel

    model.train()
    for i, batch in enumerate(dataset):
        batch_stats = train_one_batch(model, batch, loss_fn, optimizer, monitors)
        epoch_stats = {name: (epoch_stats[name] + batch_stats[name]) for name in monitors.names}

    epoch_stats = {name: stat / len(dataset) for name, stat in epoch_stats.items()}
    epoch_stats["time"] = time.time() - t0

    return epoch_stats

def evaluate_one_batch_serial(model, batch, monitors):
    batch_stats = {name: 0.0 for name in monitors.names}
    for i, G in enumerate(batch):
        # reset hidden states
        model.reset_hidden_states(G)
        # forward model
        model_output = model(G)
        # get stats
        stats = unwrap(monitors(model_output, G))
        batch_stats = {name: (batch_stats[name] + stats[name]) for name in monitors.names}

    batch_stats = {name: batch_stats[name] / len(batch) for name in monitors.names}
    return batch_stats

def evaluate_one_batch_parallel(model, batch, monitors):

    model.reset_hidden_states(batch)

    # forward model
    model_output = model(batch)

    # get stats
    batch_stats = unwrap(monitors(model_output, batch))

    return batch_stats

def evaluate_one_epoch(model, dataset, loss_fn, monitors):
    t0 = time.time()
    epoch_stats = {name: 0.0 for name in monitors.names}

    model.eval()

    for i, batch in enumerate(dataset):
        if dataset.order is None:
            batch_stats = evaluate_one_batch_serial(model, batch, monitors)
        else:
            batch_stats = evaluate_one_batch_parallel(model, batch, monitors)
        epoch_stats = {name: (epoch_stats[name] + batch_stats[name]) for name in monitors.names}

    epoch_stats = {name: stat / len(dataset) for name, stat in epoch_stats.items()}
    epoch_stats["time"] = time.time() - t0

    return epoch_stats

def get_print_function(model_dir, level):
    if level == 0:
        def _print(x):
            pass
    elif level == 1:
        def _print(x):
            with open(os.path.join(model_dir, 'log.txt'), 'a') as f:
                f.write(str(x))
    elif level == 2:
        def _print(x):
            with open(os.path.join(model_dir, 'log.txt'), 'a') as f:
                f.write(str(x))
            print(x)
    return _print
