import numpy as np
# 다음 행렬은 Eigen Value와 Eigen Vector로 diagonalize가능한가?

A = np.array([[1,1],[0,1]])
w1,V1 = np.linalg.eig(A)

print(w1)   # [1. 1.]
print(V1)
'''
[[ 1.00000000e+00 -1.00000000e+00]
 [ 0.00000000e+00  2.22044605e-16]]
'''

V1_inv = np.linalg.inv(V1)
print(V1_inv)
'''
[[1.00000000e+00 4.50359963e+15]
 [0.00000000e+00 4.50359963e+15]]
'''

print(V1 @ np.diag(w1) @ V1_inv)
'''
[[1. 0.]
 [0. 1.]]
'''