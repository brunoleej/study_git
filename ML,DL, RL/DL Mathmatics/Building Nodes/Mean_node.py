import numpy as np
from numpy.core.fromnumeric import mean

class mean_node:
    def __init__(self):
        self._x, self._y = None, None
        self._z = None

    def forward(self, x):    # forward propagation
        self._x = x
        self._z = np.mean(self._x)
        return self._z

    def backward(self, dz):
        print(len(self._x))
        print(1/len(self._x))
        print(np.ones_like(self._x))    # 1 / n을 5개를 만들어야 하기 때문에 5개를 만들어줌
        print(1 / len(self._x) * np.ones_like(self._x))
        print(dz * 1 / len(self._x) * np.ones_like(self._x))

test_x = np.random.randn(5)
print(test_x)   # [-0.50708535  0.96194217  0.13331605 -1.50446546 -0.34160774] => Normal distribution

tmp = mean_node()
z = tmp.forward(test_x)
dx = tmp.backward(2)