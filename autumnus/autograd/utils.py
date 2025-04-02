import numpy as np
from autumnus import Tensor


def tensor(data: tuple | list | np.ndarray,
           requires_grad=True) -> Tensor:
    """Returns tensor initialized from passed data."""
    data = np.array(data)
    return Tensor(data=data, requires_grad=requires_grad)

def zeros(shape: tuple) -> Tensor:
    """Returns tensor of zeros."""
    return Tensor(np.zeros(shape))

def ones(shape: tuple) -> Tensor:
    """Returns tensor of ones."""
    return Tensor(np.ones(shape))
