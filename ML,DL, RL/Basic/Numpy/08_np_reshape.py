import numpy as np

x1 = np.arange(1,16)
print(x1)   # [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
print(x1.shape) # (15,)

x2 = x1.reshape(3,5)
print(x2)
'''
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]
 [11 12 13 14 15]]
'''