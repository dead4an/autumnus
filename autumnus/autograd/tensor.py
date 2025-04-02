import numpy as np


class Tensor:
    """Tensor with autograd support."""
    def __init__(self, data: np.ndarray, operation: "Operation"=None, 
                 requires_grad=True) -> None:
        self._data = np.copy(data)
        self._operation = operation
        self._requires_grad = requires_grad

        if self._requires_grad:
            self._grad = np.zeros_like(self._data)
        else:
            self._grad = None

    def backward(self, lr: float) -> None:
        """Updates weights based on learning rate and output gradient."""
        # update weights
        self._data -= self._grad * lr

        # pass grad backward
        if self._operation:
            self._operation.backward(self._grad)

    def update_grad(self, grad) -> None:
        """Accumulates output gradient."""
        self._grad += grad

    def zero_grad(self) -> None:
        """Resets output gradient to zero."""
        self._grad = np.zeros_like(self._data)

    def __add__(self, other: "Tensor") -> "Tensor":
        op = Add()
        z = op.forward(self, other)
        return z
    
    def __sub__(self, other: "Tensor") -> "Tensor":
        op = Sub()
        z = op.forward(self, other)
        return z
    
    def __mul__(self, other: "Tensor") -> "Tensor":
        op = Mul()
        z = op.forward(self, other)
        return z
    
    def __truediv__(self, other: "Tensor") -> "Tensor":
        op = Div()
        z = op.forward(self, other)
        return z
    
    def __matmul__(self, other: "Tensor") -> "Tensor":
        op = MatMul()
        z = op.forward(self, other)
        return z

class Operation:
    """Base operation class."""
    def forward(self) -> Tensor:
        """Method used in forward pass."""
        raise NotImplementedError

    def backward(self) -> None:
        """Method used for back propagation."""
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)

class Add(Operation):
    """Addition operation."""
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        # cache operands
        self._cache = (a, b)

        # compute resulting tensor
        z_data = a._data + b._data
        requires_grad = a._requires_grad or b._requires_grad
        z = Tensor(data=z_data, operation=self, requires_grad=requires_grad)

        return z
    
    def backward(self, dz: np.array) -> None:
        # get operands from cache
        a, b = self._cache

        # compute gradients
        if a._requires_grad:
            dz_da = dz
            a.update_grad(dz_da)

        if b._requires_grad:
            dz_db = dz
            b.update_grad(dz_db)

class Sub(Operation):
    """Subtraction operation."""
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        # cache operands
        self._cache = (a, b)

        # compute resulting tensor
        z_data = a._data - b._data
        requires_grad = a._requires_grad or b._requires_grad
        z = Tensor(data=z_data, operation=self, requires_grad=requires_grad)

        return z
    
    def backward(self, dz: np.ndarray) -> None:
        # get operands from cache
        a, b = self._cache

        # compute gradients
        if a._requires_grad:
            dz_da = dz
            a.update_grad(dz_da)

        if b._requires_grad:
            dz_db = -dz
            b.update_grad(dz_db) 

class Mul(Operation):
    """Multiplication operation."""
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        # cache operands
        self._cache = (a, b)

        # compute resulting tensor
        z_data = a._data * b._data
        requires_grad = a._requires_grad
        z = Tensor(data=z_data, operation=self, requires_grad=requires_grad)

        return z
    
    def backward(self, dz: np.ndarray) -> None:
        # get operands from cache
        a, b = self._cache

        # compute gradients
        if a._requires_grad:
            dz_da = dz * b
            a.update_grad(dz_da)

        if b._requires_grad:
            dz_db = dz * a.sum()
            b.update_grad(dz_db)

class Div(Operation):
    """Division operation."""
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        # cache operands
        self._cache = (a, b)

        # compute resulting tensor
        z_data = a._data / b._data
        requires_grad = a._requires_grad or b._requires_grad
        z = Tensor(data=z_data, operation=self, requires_grad=requires_grad)

        return z
    
    def backward(self, dz: np.ndarray) -> None:
        # get operands from cache
        a, b = self._cache

        # compute gradients
        if a._requires_grad:
            dz_da = dz *  (1 / b._data)
            a.update_grad(dz_da)

        if b._requires_grad:
            dz_db = -dz * a * (1 / b**2)
            b.update_grad(dz_db)

class MatMul(Operation):
    """Matrix multiplication operation."""
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        # cache operands
        self._cache = (a, b)

        # compute resulting tensor
        z_data = a._data @ b._data
        requires_grad = a._requires_grad or b._requires_grad
        z = Tensor(data=z_data, operation=self, requires_grad=requires_grad)

        return z
    
    def backward(self, dz: np.ndarray) -> None:
        # get operands from cache
        a, b = self._cache

        # compute gradients
        if a._requires_grad:
            dz_da = dz @ b._data.T
            a.update_grad(dz_da)

        if b._requires_grad:
            dz_db = a._data.T @ dz
            b.update_grad(dz_db)
