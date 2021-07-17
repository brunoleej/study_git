import numpy as np

a = np.array([4, 2])
b = np.array([2, 8])

print(np.hstack((a, b)))    # [4 2 2 8]
print(np.column_stack((a, b)))
'''
[[4 2]
 [2 8]]
'''