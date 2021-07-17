# 연습문제 2.1.3
# 1. Numpy를 사용해 붓꽃 데이터 X의 전치행렬 X_transpose를 구한다.
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()  # Data load
# X = iris.data[0,:]  # 첫번째 꽃의 데이터
X = np.array([[5.1],[3.5],[1.4],[0.2]])

# print(X)    # [5.1 3.5 1.4 0.2]
# print(X.shape)  # (4,)
# print(X.ndim)   # 1
print(X)
'''
[[5.1]
 [3.5]
 [1.4]
 [0.2]]
'''
print(X.shape)  # (4, 1)
print(X.ndim)   # 2

X_tran = X.T    
# print(X_tran)    # [5.1 3.5 1.4 0.2]
print(X_tran)   # [[5.1 3.5 1.4 0.2]]

# 2. Numpy를 사용해서 위 전치행렬을 다시 전치한 행렬(X_transpose)transpose를 구한다. 이 행렬과 원래 행렬 X를 비교한다.
X_tran_tran = X_tran.T
print(X_tran_tran)
'''
[[5.1]
 [3.5]
 [1.4]
 [0.2]]
'''