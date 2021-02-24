import numpy as np
# Numpy 에서의 Qudratic Form

X = np.array([1,2,3])
print(X)    # [1 2 3]

A = np.arange(1,10).reshape(3,3)
print(A)   
'''
[[1 2 3]
 [4 5 6]
 [7 8 9]]
'''

print(X.T @ A @ X)  # 228
