import numpy as np

X1 = np.array([[1,3],[2,4]])
print(np.linalg.matrix_rank(X1))    # 2

X2 = np.array([[1,3,5],[2,3,7]])
print(np.linalg.matrix_rank(X2))    # 2

