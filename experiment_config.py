
from collections import namedtuple
from monitors import classification_monitors, regression_monitors
from models import get_model_generator, get_model_config
import torch.nn as nn
from losses import CrossEntropy, MSEGraphLoss

ExperimentConfig = namedtuple(
        'ExperimentConfig', [
            'model_generator',
            'model_config',
            'mode',
            'loss_fn',
            'monitors'
        ]
)

def get_experiment_config(args, dataset):
    model_generator=get_model_generator(args.model)
    model_config = get_model_config(args.model, args, dataset)
    mode=dataset.problem_type

    if mode == 'clf': # classification
        loss_fn = CrossEntropy(
            graph_targets=dataset.graph_targets,
            )
        monitors = classification_monitors(args, dataset)
    elif mode == 'reg': # regression
        loss_fn = MSEGraphLoss(
            graph_targets=dataset.graph_targets,
            )
        monitors = regression_monitors(args, dataset)

    config = ExperimentConfig(
        model_generator=model_generator,
        model_config=model_config,
        mode=mode,
        loss_fn=loss_fn,
        monitors=monitors
    )
    return config
