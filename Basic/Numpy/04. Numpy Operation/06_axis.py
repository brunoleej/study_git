import numpy as np

a = np.arange(12).reshape(3,4)
print(a)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
'''

print(a.sum(axis = 0))      # [12 15 18 21]
print(a.min(axis = 1))      # [0 4 8]
print(a.max(axis = 1))      # [ 3  7 11]