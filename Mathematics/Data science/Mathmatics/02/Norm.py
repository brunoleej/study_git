import numpy as np

A = (np.arange(9) -4).reshape((3, 3))
print(A)
'''
[[-4 -3 -2]
 [-1  0  1]
 [ 2  3  4]]
'''

print(np.linalg.norm(A))    # 7.745966692414834



