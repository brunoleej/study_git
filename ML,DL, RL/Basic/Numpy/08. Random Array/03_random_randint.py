import numpy as np

a = np.random.randint(2, size=5)
print(a)    # [0 1 0 0 0]

b = np.random.randint(2, 4, size=5)
print(b)    # [2 3 3 2 2]

c = np.random.randint(1, 5, size=(2, 3))
print(c)   
'''
[[4 4 3]
 [1 4 3]]
'''