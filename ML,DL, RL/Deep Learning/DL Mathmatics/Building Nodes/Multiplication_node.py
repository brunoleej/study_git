import numpy as np

class plus_Node:
    def __init__(self):
        self._x, self._y = None, None
        self._z = None

    def forward(self, x, y):    # forward propagation
        self._x, self._y = x, y
        self._z = self._x * self._y
        return self._z

    def backward(self, dz):
        return dz * self._y, dz * self._x

test_x = np.random.randn(5)
print(test_x)