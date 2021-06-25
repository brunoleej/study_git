import numpy as np

class plus_Node:
    def __init__(self):
        self._x = None
        self._z = None

    def forward(self, x):    # forward propagation
        self._x = x
        self._z = self._x * self._x
        return self._z

    def backward(self, dz):
        return dz * 2 * self._x

test_x = np.random.randn(5)
print(test_x)
