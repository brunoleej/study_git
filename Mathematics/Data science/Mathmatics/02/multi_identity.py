import numpy as np

# AI = IA = A
# 항등행렬(identity matrix)의 곱셈
A = np.array([[1,2],[3,4]])
I = np.eye(2)

print(A @ I)
'''
[[1. 2.]
 [3. 4.]]
'''
print(I @ A)
'''
[[1. 2.]
 [3. 4.]]
'''