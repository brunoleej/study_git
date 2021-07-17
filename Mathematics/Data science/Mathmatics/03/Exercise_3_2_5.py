import numpy as np

# 1. 다음 벡터로 랭크-1 행렬을 만들고 Numpy로 랭크를 계산하여 실제로 1이 나오는지 확인하라
x_1 = np.array([[1],[2]])
x_2 = np.array([[2],[4]])

M = x_1 @ x_2.T

print(M)
'''
[[2 4]
 [4 8]]
'''

print(np.linalg.matrix_rank(M)) # 1

# 2. 다음 두개의 벡터로 랭크-2 행렬을 만들고 numpy로 계산해 2가 나오는지 확인하라.
x_1 = np.array([[1],[1]])
x_2 = np.array([[1],[-1]])
x_1 = 1 * x_1
x_2 = 2 * x_2

M = x_1 @ x_2.T

print(M)
'''
[[ 1 -1]
 [ 1 -1]]
'''
print(np.linalg.matrix_rank(M))