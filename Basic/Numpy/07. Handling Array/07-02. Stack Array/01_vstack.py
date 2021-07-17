import numpy as np

a = np.floor(10*np.random.random((2, 2)))
b = np.floor(10*np.random.random((2, 2)))

print(a)
'''
[[7. 1.]
 [9. 4.]]
'''
print(b)
'''
[[6. 4.]
 [9. 0.]]
'''
print(np.vstack((a, b)))
'''
[[7. 1.]
 [9. 4.]
 [6. 4.]
 [9. 0.]]
'''