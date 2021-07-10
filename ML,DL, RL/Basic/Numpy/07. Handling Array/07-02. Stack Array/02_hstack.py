import numpy as np

a = np.floor(10*np.random.random((2, 2)))
b = np.floor(10*np.random.random((2, 2)))

print(a)
'''
[[6. 1.]
 [1. 4.]]
'''
print(b)
'''
[[8. 6.]
 [2. 7.]]
'''
print(np.hstack((a, b)))
'''
[[6. 1. 8. 6.]
 [1. 4. 2. 7.]]
'''