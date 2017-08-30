
"""
train.py

Core training script for the addition task-specific NPI. Instantiates a model, then trains using
the precomputed data.
"""
import torch
from torch.autograd import Variable
import numpy as np
import time
