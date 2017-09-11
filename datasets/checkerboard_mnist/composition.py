import numpy as np

def swiss_roll(x):
    # batched
    return np.stack((x[:, 0] * np.cos(x[:, 0]), x[:, 1] * np.sin(x[:, 1])), 1)

def f_1(x):
    return x[:, 0] ** 2 * x[:, 1] - x[:, 0] / (1 + x[:, 1] ** 2)

def f_2(x):
    return np.sum(x, 1) / np.sum(x ** 2, 1)
