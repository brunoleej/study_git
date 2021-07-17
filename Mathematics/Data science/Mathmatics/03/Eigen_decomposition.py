import numpy as np

# Eigen Vector는 열로 보면 1개의 벡터임
A = np.array([[1, -2], [2, -3]])
w1, V1 = np.linalg.eig(A)

print(w1)   # [-0.99999998 -1.00000002] --> Eigen Value
print(V1)   # --> Eigen vector
'''
[[0.70710678 0.70710678]
 [0.70710678 0.70710678]]
'''

B = np.array([[2, 3], [2, 1]])
w2, V2 = np.linalg.eig(B)

print(w2)   # [ 4. -1.] --> Eigen Value
print(V2)   # --> Eigen Value
'''
[[ 0.83205029 -0.70710678]
 [ 0.5547002   0.70710678]]
'''

C = np.array([[0, -1], [1, 0]])
w3, V3 = np.linalg.eig(C)

print(w3)   # [0.+1.j 0.-1.j] --> Eigen Value
print(V3)   # --> Eigen Value
'''
[[0.70710678+0.j         0.70710678-0.j        ]
 [0.        -0.70710678j 0.        +0.70710678j]]
'''