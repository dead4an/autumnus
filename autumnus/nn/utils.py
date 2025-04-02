import numpy as np


def xavier_normal(input_dim: int, output_dim: int):
    """Xavier normal initializer."""
    mean = 0
    variance = np.sqrt(2 / (input_dim + output_dim))
    return np.random.normal(mean, variance, size=(input_dim, output_dim))
