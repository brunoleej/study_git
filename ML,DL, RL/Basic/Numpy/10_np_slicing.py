import numpy as np

x1 = np.arange(10)
print(x1)   # [0 1 2 3 4 5 6 7 8 9]

x1[3] = 100
print(x1)   # [  0   1   2 100   4   5   6   7   8   9]


# 2 dimension
x2 = np.arange(10).reshape(2,5)
print(x2)
'''
[[0 1 2 3 4]
 [5 6 7 8 9]]
'''

print(x2[-1,1]) # 6
print(x2[0])    # [0 1 2 3 4]

# slicing
x3 = np.arange(10)
print(x3)   # [0 1 2 3 4 5 6 7 8 9]
print(x3[1:])   # [1 2 3 4 5 6 7 8 9]

x4 = np.arange(10).reshape(2,5)
print(x4)
'''
[[0 1 2 3 4]
 [5 6 7 8 9]]
'''
print(x4[0, :2])    # [0 1]
print(x4[:1, :2])   # [[0 1]]

# 3 dimension
x5 = np.arange(54).reshape(2,9,3)
print(x5)
print(x5[:1, :2, :])
'''
[[[0 1 2]
  [3 4 5]]]
'''
print(x5[0, :2, :])
'''
[[0 1 2]
 [3 4 5]]
'''