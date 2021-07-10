import numpy as np

np.random.seed(0)

a = np.floor(10*np.random.random((2, 2)))
b = np.floor(10*np.random.random((2, 2)))

print(np.hstack((a, b)))
'''
[[5. 7. 4. 6.]
 [6. 5. 4. 8.]]
'''
print(np.column_stack((a, b)))
'''
[[5. 7. 4. 6.]
 [6. 5. 4. 8.]]
'''