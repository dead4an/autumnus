import numpy as np
import autumnus as atm
import autumnus.nn as nn


def test_linear():
    atm.seed(42) # fix seed for weights initialization
    input_tensor = atm.tensor(((1, 1, 1), (2, 2, 2)))
    fc1 = nn.Linear(input_dim=3, output_dim=6)
    fc2 = nn.Linear(input_dim=6, output_dim=1)

    out1 = fc1(input_tensor)
    out2 = fc2(out1)

    assert np.array_equal(out1._data, 
                          input_tensor._data @ fc1._weights._data + fc1._bias._data)
    
    assert np.array_equal(out2._data, 
                          out1._data @ fc2._weights._data + fc2._bias._data)
