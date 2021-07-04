# Numpy object is homogeneous multidimensional array.
# ndarray.ndim : dimension of array
# ndarray.shape : shape of array
# ndarray.size : elments of array
# ndarray.dtype : type of array
# ndarray.itemsize : size of array
# ndarray.data : real elements of array

import numpy as np

a = np.arange(15).reshape(3, 5)
print(a)
'''
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
'''
print('a.shape:', a.shape)              # a.shape: (3, 5)
print('a.ndim:', a.ndim)                # a.ndim: 2
print('a.dtype.name:', a.dtype.name)    # a.dtype.name: int32
print('a.itemsize:', a.itemsize)        # a.itemsize: 4
print('a.size:', a.size)                # a.size: 15        
print('type(a):', type(a))              # type(a): <class 'numpy.ndarray'>