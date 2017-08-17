
from collections import namedtuple
from monitors import CLF_MONITORS, REG_MONITORS
from models import get_model_generator, get_model_config
import torch.nn as nn

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
        loss_fn = nn.BCELoss()
        monitors = CLF_MONITORS
    elif mode == 'reg': # regression
        loss_fn = nn.MSELoss()
        monitors = REG_MONITORS

    config = ExperimentConfig(
        model_generator=model_generator,
        model_config=model_config,
        mode=mode,
        loss_fn=loss_fn,
        monitors=monitors
    )
    return config
