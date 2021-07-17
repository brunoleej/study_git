import numpy as np
import scipy as sp
import scipy.optimize

A = np.array([[-1, 0], [0, -1], [1, 2], [4, 5]])
b = np.array([-100, -100, 500, 9800])
c = np.array([-3, -5])

result = sp.optimize.linprog(c, A, b)
print(result)
