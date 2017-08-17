
"""
train.py

Core training script for the addition task-specific NPI. Instantiates a model, then trains using
the precomputed data.
"""
import torch
from torch.autograd import Variable
import numpy as np
import time

def evaluate(model, dataset, loss_fn, monitors):
    eval_stats = {name: 0.0 for name in monitors.keys()}

    t0 = time.time()
    for i, G in enumerate(dataset):
        # reset hidden states
        G = model.reset_hidden_states(G)

        # forward model
        readout = model(G)

        # stats
        stats = {name: monitor(G, readout) for name, monitor in monitors.items()}
        eval_stats = {name: (eval_stats[name] + stats[name]) for name in monitors.keys()}

    t = time.time() - t0
    eval_stats = {name: stat / len(dataset) for name, stat in eval_stats.items()}
    eval_stats["time"] = t
    return eval_stats
