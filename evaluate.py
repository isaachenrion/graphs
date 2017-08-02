
"""
train.py

Core training script for the addition task-specific NPI. Instantiates a model, then trains using
the precomputed data.
"""
import torch
from torch.autograd import Variable
import numpy as np
import time
from mpnn import MPNN

def evaluate(model, examples, loss_fn):
    eval_loss, eval_acc = 0., 0.
    t0 = time.time()
    for i, G in enumerate(examples):
        # reset hidden states
        for u in G.nodes():
            G.node[u]['hidden'] = Variable(torch.zeros(1, model.state_dim))
            G.node[u]['message'] = Variable(torch.zeros(1, model.message_dim))

        # forward model
        readout = model(G)

        # get loss
        loss = loss_fn(readout, G.graph['readout'])

        # stats
        hard_prediction = readout.data.numpy()[0,0] > 0.5
        ground_truth = G.graph['readout'].data.numpy()[0,0]
        eval_acc += (hard_prediction == ground_truth)

        eval_loss += loss.data.numpy()[0]

    t = time.time() - t0
    return {"loss": eval_loss / len(examples), "acc":eval_acc/len(examples), "t": t}
