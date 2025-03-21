import numpy as np


class Tensor:
    """Tensor class with autograd support."""
    def __init__(self, data: np.ndarray, operation: "Operation"=None, requires_grad=True):
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
    def forward(self) -> Tensor:
        """Method used for forward propagation."""
        raise NotImplementedError

    def backward(self) -> None:
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

        return z
    
    def backward(self, dz: np.array) -> None:
        # get operands from cache
        a, b = self._cache

        # compute gradients
        if a._requires_grad:
            dz_da = np.ones_like(a._data)
            a.update_grad(dz_da)

        if b._requires_grad:
            dz_db = np.ones_like(b._data)
            b.update_grad(dz_db)


class Sub(Operation):
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
            dz_da = np.ones_like(a._data)
            a.update_grad(dz_da)

        if b._requires_grad:
            dz_db = -np.ones_like(b._data)
            b.update_grad(dz_db)
    

class Mul(Operation):
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
            dz_da = np.full_like(a._data, b)
            a.update_grad(dz_da)


class MatMul(Operation):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        # cache operands
        self._cache = (a, b)

        # compute resulting tensor
        z_data = a._data @ b._data
        requires_grad = a._requires_grad or b._requires_grad
        z = Tensor(data=z_data, operation=self, requires_grad=requires_grad)

        return z
    
    def backward(self, dz: np.ndarray) -> None:
        """
            a11 a12 a1n
        A = a21 a22 a2n
            am1 am2 amn
            
            b11 b12 b1k
        B = b21 b22 b2k
            bn1 bn2 bnk
            
            a11b11 + ... + a1nbn1
        Y = a21b11 + ... + a2nbn1
            am1b11 + ... + amnbn1
            
                dL_dy11             dL_dy11 * dy11_da11 ... dL_dy11 * dy11_da1n    
        dL_dY = dL_dy21     dL_dA = dL_dy21 * dy21_da21 ... dL_dy21 * dy21_da2n = 
                dL_dym1             dL_dym1 * dym1_dam1 ... dL_dym1 * dym1_damn

                                    dL_dy11 * b11 ... dL_dy11 * bn1
                                  = dL_dy21 * b11 ... dL_dy21 * bn1 = dL_dY * B^T
                                    dL_dym1 * b11 ... dL_dym1 * bn1

                dL_dy11 * dy11_db11 + ... + dL_dym1 * dym1_db11
        dL_dB = dL_dy11 * dy11_db21 + ... + dL_dym1 * dym1_db21 =
                dL_dy11 * dy11_dbn1 + ... + dL_dym1 * dym1_dbn1

                dL_dy11 * a11 + ... + dL_ym1 * am1
              = dL_dy11 * a12 + ... + dL_ym1 * am2 = A^T * dL_dY
                dL_dy11 * a1n + ... + dL_ym1 * amn
                """
        # get operands from cache
        a, b = self._cache

        # compute gradients
        if a._requires_grad:
            dz_da = dz @ b._data.T
            a.update_grad(dz_da)

        if b._requires_grad:
            dz_db = a._data.T @ dz
            b.update_grad(dz_db)
