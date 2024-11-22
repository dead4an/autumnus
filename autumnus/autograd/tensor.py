import numpy as np


class Tensor:
    """Tensor class compatible with autograd operations.
    
    :parameter data: Data array the Tensor to contain.
    :parameter operation: Operation object created this Tensor.
    :parameter requires_grad: Whether autograd to compute gradients 
    for the Tensor during backpropagation.
    """
    def __init__(self, data, operation=None, requires_grad=True) -> None:
        self.data = data
        self.operation = operation
        self.requires_grad = requires_grad
        self.shape = self.data.shape
        self.grad = None

        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
    def backward(self, output_gradient) -> None:
        """Method used in backpropagation.
        Computes gradient for the Tensor based on operation
        and resulting Tensor.

        :parameter output: Resulting Tensor of operation on this Tensor.
        """
        # Add output gradient to the Tensor
        self.grad += output_gradient

        # Pass gradient backward
        if self.operation is not None:
            self.operation.backward(self.grad)

    # Arithmetic operations
    def __add__(self, other) -> "Tensor":
        operation = Add()
        result = operation.forward(self, other)
        
        return result
    
    def __radd__(self, other) -> "Tensor":
        operation = Add()
        result = operation.forward(self, other)
        
        return result
    
    def __iadd__(self, other) -> "Tensor":
        operation = Add()
        result = operation.forward(self, other)
        
        return result

    def __neg__(self) -> "Tensor":
        operation = Neg()
        result = operation.forward(self)

        return result

    def __sub__(self, other) -> "Tensor":
        # addition and negation are defined above
        result = self + (-other)
        
        return result 
    
    def __mul__(self, other) -> "Tensor":
        operation = Mul()
        result = operation.forward(self, other)

        return result
    
    def __rmul__(self, other) -> "Tensor":
        operation = Mul()
        result = operation.forward(self, other)

        return result
    
    def __imul__(self, other) -> "Tensor":
        operation = MatMul()
        result = operation.forward(self, other)

        return result

    def __matmul__(self, other) -> "Tensor":
        operation = MatMul()
        result = operation.forward(self, other)

        return result

    def __rmatmul__(self, other) -> "Tensor":
        operation = MatMul()
        result = operation.forward(self, other)

        return result

    def __imatmul__(self, other) -> "Tensor":
        operation = Mul()
        result = operation.forward(self, other)

        return result

# Operations
class Operation:
    """Base operation class.
    Operates on Tensors and keeps track on computation graph."""
    def forward(self, *input) -> Tensor:
        """Method used during forward pass.
        Takes input Tensor(s), performs an operation and returns
        a resulting Tensor.
        
        :parameter input: Input Tensor.
        """

        raise NotImplementedError
    
    def backward(self, output_gradient) -> None:
        """Method used during backpropagation.
        Takes output Tensor's gradient and computes gradients for 
        operand Tensor(s) using the chain rule.
        
        :parameter output_gradient: Output (resulting) Tensor's gradient.
        """

        raise NotImplementedError
    

class Add(Operation):
    """Addition operation."""
    def forward(self, a, b) -> Tensor:
        """Method used during forward pass.
        
        :parameter a: First Tensor.
        :parameter b: Second Tensor.
        """
        # Save operands to cache
        self._cache = (a, b)

        # Perform operation
        z_data = a.data + b.data
        requires_grad = a.requires_grad or b.requires_grad

        # Initialize resulting Tensor
        z = Tensor(data=z_data, operation=self, requires_grad=requires_grad)

        return z
    
    def backward(self, dz) -> None:
        """Method used during backpropagation.
        Computes gradients using the chain rule.

        :parameter dz: Output (resulting) Tensor's gradient.
        """
        # Get operands from cache
        a, b = self._cache
        
        # Compute gradient for "a"
        if a.requires_grad:
            # d/da(a + b) = 1
            da = dz
            a.backward(da)

        # Compute gradient for "b"
        if b.requires_grad:
            # d/db(a + b) = 1
            db = dz
            b.backward(db)


class Neg(Operation):
    """Negation operation."""
    def forward(self, a) -> Tensor:
        """Method used during forward pass
        
        :parameter a: Tensor to be negated.
        """
        # Save operand to cache
        self._cache = a

        # Perform operation
        z_data = -a.data
        requires_grad = a.requires_grad

        # Initialize resulting Tensor
        z = Tensor(data=z_data, operation=self, requires_grad=requires_grad)

        return z

    def backward(self, dz) -> None:
        """Method used during backpropagation.
        Computes gradients using the chain rule.
        
        :parameter dz: Output (resulting) Tensor's gradient.
        """
        # Get operand from cache
        a = self._cache

        # Compute gradient for "a"
        if a.requires_grad:
            # d/da(-a) = -1
            da = dz * -1
            a.backward(da)


class Mul(Operation):
    """Multiplication operation."""
    def forward(self, a, b) -> Tensor:
        """Method used during forward pass.
        
        :parameter a: First Tensor.
        :parameter b: Second Tensor.
        """
        # Save operands to cache
        self._cache = (a, b)

        # Perform operation
        z_data = a.data * b.data
        requires_grad = a.requires_grad or b.requires_grad

        # Initialize resulting Tensor
        z = Tensor(data=z_data, operation=self, requires_grad=requires_grad)

        return z

    def backward(self, dz) -> None:
        """Method used during backpropagation.
        Computes gradients using the chain rule.

        :parameter dz: Output (resulting) Tensor's gradient.
        """
        # Get operands from cache
        a, b = self._cache

        # Compute gradient for "a"
        if a.requires_grad:
            # d/da(a * b) = b
            da = dz * b.data
            a.backward(da)

        # Compute gradient for "b"
        if b.requires_grad:
            # d/db(a * b) = a
            db = dz * a.data
            b.backward(db)


class Div(Operation):
    """Division operation."""
    def forward(self, a, b) -> Tensor:
        """Method used during forward pass.
        
        :parameter a: First Tensor (dividend).
        :parameter b: Second Tensor (divisor).
        """
        # Save operands to cache
        self._cache = (a, b)

        # Perform operation
        z_data = a.data / b.data
        requires_grad = a.requires_grad or b.requires_grad

        # Initialize resulting Tensor
        z = Tensor(data=z_data, operation=self, requires_grad=requires_grad)

        return z

    def backward(self, dz) -> None:
        """Method used during backpropagation.
        Computes gradients using the chain rule.

        :parameter dz: Output (resulting) Tensor's gradient.
        """
        # Get operands from cache
        a, b = self._cache

        # Compute gradient for "a"
        if a.requires_gradient:
            # d/da(a / b) = b / b^2
            da = dz * (1 / b.data)
            a.backward(da)

        # Compute gradient for "b"
        if b.requires_gradient:
            # d/db(a / b) = -(a / b^2)
            db = dz * -(a.data / b.data**2)
            b.backward(db)


class MatMul(Operation):
    """Matrix multiplication"""
    def forward(self, a, b) -> Tensor:
        """Method used during forward pass.
        
        :parameter a: First Tensor (left).
        :parameter b: Second Tensor (right).
        """
        # Save operands to cache
        self._cache = (a, b)

        # Perform operation
        z_data = a.data @ b.data
        requires_grad = a.requires_grad or b.requires_grad

        # Initialize resulting Tensor
        z = Tensor(data=z_data, operation=self, requires_grad=requires_grad)

        return z

    def backward(self, dz) -> None:
        """Method used during backpropagation.
        Computes gradients using the chain rule.
        
        :parameter dz: Output (resulting) Tensor's gradient.
        """
        # Get operands from cache
        a, b = self._cache

        # Compute gradient for "a"
        if a.requires_gradient:
            # d/da(a @ b) = bT
            da = dz @ b.data.T
            a.backward(da)

        # Compute gradient for "b"
        if b.requires_gradient:
            # d/db(a @ b) = aT
            db = a.data.T @ dz
            b.backward(db)
