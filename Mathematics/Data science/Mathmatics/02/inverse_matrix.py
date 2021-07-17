import numpy as np

A = np.array([[1,1,0],[0,1,1],[1,1,1]])
print(A)
'''
[[1 1 0]
 [0 1 1]
 [1 1 1]]
'''

# 역행렬(inverse_matrix 계산)
Ainv = np.linalg.inv(A)
print(Ainv)
'''
[[ 0. -1.  1.]
 [ 1.  1. -1.]
 [-1.  0.  1.]]
'''
