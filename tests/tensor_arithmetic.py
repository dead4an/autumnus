import pytest
import numpy as np
import autumnus as atm


def test_add():
    a = np.array([[0, 1, 2], [-2, -1, 0], [1.2, -1.3, 0.23]])
    b = np.array([[1, 2, 3], [-1, -2, -3], [-1.4, 1.6, 0.21]])
    a = atm.tensor(a)
    b = atm.tensor(b)
    c = a + b
    assert np.array_equal(c._data, a._data + b._data)

def test_sub():
    a = np.array([[0, 1, 2], [-2, -1, 0], [1.32, -1.6, 0.1]])
    b = np.array([[1, 2, 3], [-1, -2, -3], [-1.98, 1.5, 0.2]])
    a = atm.tensor(a)
    b = atm.tensor(b)
    c = a - b
    assert np.array_equal(c._data, a._data - b._data)

def test_mul():
    a = np.array([[0, 0, 0], [1, 1, 1], [-2.3, 1.15, 0.1]])
    b = np.array([[-1, 0, 1], [1, 2, -3], [-1.7, -2.3, 0.001]])
    a = atm.tensor(a)
    b = atm.tensor(b)
    c = a * b
    assert np.array_equal(c._data, a._data * b._data)

def test_div():
    a = np.array([[0, 1, -1], [-1, 2, -3], [0.23, 0.001, -2.01]])
    b = np.array([[1, 1, -1], [-7, 3, -2], [-0.15, 1.001, 0.001]])
    a = atm.tensor(a)
    b = atm.tensor(b)
    c = a / b
    assert np.array_equal(c._data, a._data / b._data)

def test_zero_div():
    a = np.array([1])
    b = np.array([0])
    a = atm.tensor(a)
    b = atm.tensor(b)
    
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        c = a / b
        assert c._data == np.array([np.inf])

def test_matmul():
    a = np.array([[0, 1, 2], [-1, -2, 0], [0.1, 0.5, -0.1]])
    b = np.array([[0, 1, 0], [-2, 4, 0], [-10, 0.2, -0.5]])
    a = atm.tensor(a)
    b = atm.tensor(b)
    c = a @ b
    assert np.array_equal(c._data, a._data @ b._data)
