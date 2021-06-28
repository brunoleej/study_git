import numpy as np

x = np.arange(15)
print(x)    # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]

print(np.sum(x, axis=0))    # 105
# print(np.sum(x, axis=1)) ==> error

# matrix
y = x.reshape(3,5)
print(y)
'''
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
'''

print(np.sum(y, axis=0))    # [15 18 21 24 27]
print(np.sum(y, axis = 1))  # [10 35 60]

# Tensor
z = np.arange(36).reshape(3, 4, 3)
print(z)

print(np.sum(z, axis = 1))
print(np.sum(z, axis = 2))
print(np.sum(z, axis = -3))

# if axis is tuple
print(z)

print(np.sum(z, axis = (0,2)))