import numpy as np

a = np.arange(6)                         # 1d array
print(a)                                 # [0 1 2 3 4 5]

b = np.arange(12).reshape(4,3)           # 2d array
print(b)
'''
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
'''

c = np.arange(24).reshape(2,3,4)         # 3d array
print(c)
'''
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
'''