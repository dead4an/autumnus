import numpy as np


class Tensor:
    """Tensor class with autograd support."""
    def __init__(self, data: np.array, operation: "Operation"=None, requires_grad=True):
        self._data = np.copy(data)
        self._operation = operation
        self._requires_grad = requires_grad

        if self._requires_grad:
            self._grad = np.zeros_like(self._data)
        else:
            self._grad = None

    def backward(self):
        """Method used for back propagation."""
        ...

    def update_grad(self, grad):
        self._grad += grad

    def zero_grad(self):
        self._grad = np.zeros_like(self._data)

    def __add__(self, other: "Tensor") -> "Tensor":
        op = Add()
        z = op.forward(self, other)
        return z
    
    def __sub__(self, other) -> "Tensor":
        op = Sub(self, other)
        z = op.forward(self, other)
        return z
    
    def __mul__(self, other) -> "Tensor":
        op = Mul(self, other)
        z = op.forward(self, other)
        return z


class Operation:
    """Base operation class."""
    def forward(self):
        """Method used for forward propagation."""
        raise NotImplementedError

    def backward(self):
        """Method used for back propagation."""
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)


class Add(Operation):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        # cache operands
        self._cache = (a, b)

        # compute resulting tensor
        z_data = a._data + b._data
        requires_grad = a._requires_grad or b._requires_grad
        z = Tensor(data=z_data, operation=self, requires_grad=requires_grad)

        # compute gradients
        if a._requires_grad:
            dz_da = np.ones_like(a._data)
            a.update_grad(dz_da)

        if b._requires_grad:
            dz_db = np.ones_like(b._data)
            b.update_grad(dz_db)

        return z


class Sub(Operation):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        # cache operands
        self._cache = (a, b)

        # compute resulting tensor
        z_data = a._data - b._data
        requires_grad = a._requires_grad or b._requires_grad
        z = Tensor(data=z_data, operation=self, requires_grad=requires_grad)

        # compute gradients
        if a._requires_grad:
            dz_da = np.ones_like(a._data)
            a.update_grad(dz_da)

        if b._requires_grad:
            dz_db = -np.ones_like(b._data)
            b.update_grad(dz_db)

        return z
    

class Mul(Operation):
    def forward(self, a: Tensor, b: float) -> Tensor:
        # cache operands
        self._cache = (a, b)

        # compute resulting tensor
        z_data = a._data * b
        requires_grad = a._requires_grad
        z = Tensor(data=z_data, operation=self, requires_grad=requires_grad)

        # compute gradients
        if a._requires_grad:
            dz_da = np.full_like(a._data, b)
            a.update_grad(dz_da)

        return z