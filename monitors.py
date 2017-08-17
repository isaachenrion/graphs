import torch
import torch.nn.functional as F


def accuracy(G, readout):
    hard_prediction = readout > 0.5
    ground_truth = G.graph['readout']
    acc = (hard_prediction == ground_truth) * 100
    acc = torch.mean(acc).data.numpy()
    return acc

def bce(G, readout):
    bce = F.binary_cross_entropy(readout, G.graph['readout'])
    bce = bce.data.numpy()[0]
    return bce

def mse(G, readout):
    mse = torch.mean(torch.pow(readout - G.graph['readout'], 2))
    mse = mse.data.numpy()[0]
    return mse

def mae(G, readout):
    mae = torch.mean(torch.abs(readout - G.graph['readout']))
    mae = mae.data.numpy()[0]
    return mae


CLF_MONITORS = {"loss": bce, "accuracy": accuracy}
REG_MONITORS = {"loss": mse, "MAE": mae}
