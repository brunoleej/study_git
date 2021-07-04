import numpy as np

a = np.array([1, 2, 3, 4])
print(a)        # [1 2 3 4]
print(type(a))  # <class 'numpy.ndarray'>
print(a.dtype)  # int32

b = np.array([1.2, 3.5, 5.1])   
print(b)        # [1.2 3.5 5.1]
print(type(b))  # <class 'numpy.ndarray'>
print(b.dtype)  # float64

# a = np.array(1,2,3,4)    # Wrong
b = np.array([1,2,3,4])  # Right
c = np.array((1,2,3,4))  # Right

# 2 dimension Array
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
'''
[[1 2 3]
 [4 5 6]]
'''

