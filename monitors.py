import torch
import torch.nn.functional as F
from losses import *

def classification_monitors(args, dataset):
    monitors = LossCollection(
        primary_loss=CrossEntropy(
            graph_targets=dataset.graph_targets,
            as_dict=False
            ),
        other_losses=[
            Accuracy(
                graph_targets=dataset.graph_targets,
                as_dict=True
            )
        ]
    )
    return monitors

def regression_monitors(args, dataset):
    monitors = LossCollection(
        primary_loss=MSEGraphLoss(
            graph_targets=dataset.graph_targets,
            as_dict=False
            ),
        other_losses=[
            MAEGraphLoss(
                graph_targets=dataset.graph_targets,
                as_dict=True
            ),

        ]
    )
    return monitors
