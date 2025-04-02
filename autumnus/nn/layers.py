import numpy as np

from autumnus.autograd.tensor import Tensor
from autumnus.autograd.utils import zeros
from autumnus.nn.utils import xavier_normal


class BaseLayer:
    """Base layer class."""
    def forward(self) -> Tensor:
        """Method used in forward pass."""
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    
class Linear(BaseLayer):
    """Linear layer."""
    def __init__(self, input_dim: int, output_dim: int, requires_grad=True) -> None:
        self._weights = Tensor(xavier_normal(input_dim, output_dim))
        self._bias = zeros(shape=output_dim)

    def forward(self, input_tensor: Tensor) -> Tensor:
        z = input_tensor @ self._weights + self._bias
        return z
