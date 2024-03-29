import numpy as np

a = np.random.randn(5)
print(a)    # [ 0.83579414  0.07528817  0.40011052  0.08054437 -0.23438913]

b = np.random.randn(2, 3)
print(b)
'''
[[ 0.81178445  1.56168502  0.79348462]
 [ 1.00842402 -0.24142733 -0.00475967]]
'''

sigma, mu = 1.5, 2.0

c = sigma * np.random.randn(5) + mu
print(c)    # [1.87764762 2.51142611 0.18131304 2.54907625 1.75012746]