import numpy as np
# 다음 행렬을 고윳값과 고유 벡터로 대각화하라.
A = np.array([[2,3],[2,1]])
w1, V1 = np.linalg.eig(A)

print(w1)   # [ 4. -1.]
print(V1)
'''
[[ 0.83205029 -0.70710678]
 [ 0.5547002   0.70710678]]
'''

V1_inv = np.linalg.inv(V1)
print(V1_inv)
'''
[[ 0.72111026  0.72111026]
 [-0.56568542  0.84852814]]
'''

print(V1 @ np.diag(w1) @ V1_inv)
'''
[[2. 3.]
 [2. 1.]]
'''