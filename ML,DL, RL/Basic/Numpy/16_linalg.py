import numpy as np

x = np.random.rand(3, 3, 3)
print(x)

print(np.matmul(x, np.linalg.inv(x)))
print(x @ np.linalg.inv(x))

A = np.array([[1, 1], [2, 4]])
B = np.array([25, 64])

x = np.linalg.solve(A, B)
print(x)    # [18.  7.]

print(np.allclose(A@x, B))  # True