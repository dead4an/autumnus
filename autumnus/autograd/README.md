# ⚙️ Autograd (Automatic Differentiation)

Autograd module is designed to store data in tensors, track operations, and compute gradients during backpropagation (automatic differentiation).

## 🔢 Tensor and operations

Contains Tensor class and its arithmetic operations definition.

Each operation has forward and backward methods. During a forward pass, the operation object saves operand tensors and calculates the resulting tensor. Operation is then stored as an attribute of the result tensor. During a backward pass, it computes gradients for operands based on the result tensor gradients and operands data using the chain rule.
