import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

data = load_digits()
X = data.data
print(X)
# 첫번째 행이 첫번째 이미지를 풀어놓은것이며 2번째, 3번째 ...
'''
[[ 0.  0.  5. ...  0.  0.  0.]
 [ 0.  0.  0. ... 10.  0.  0.]
 [ 0.  0.  0. ... 16.  9.  0.]
 ...
 [ 0.  0.  1. ...  6.  0.  0.]
 [ 0.  0.  2. ... 12.  0.  0.]
 [ 0.  0. 10. ... 12.  1.  0.]]
'''

X1 = X[0,:]
X10 = X[9, :]

print(X1 @ X10) # 2807.0

