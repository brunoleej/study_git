import numpy as np

# 배열 생성
x = np.array([10, 11, 12, 13, 14])
y = np.array([0, 1, 2, 3, 4])

# 덧셈
print(x + y)    # [10 12 14 16 18]
print(x - y)    # [10 10 10 10 10]

# 행렬 덧셈
print(np.array([[5,6],[7,98]]) + np.array([[10,20],[30,40]]) - np.array([[1,2],[3,4]]))
'''
[[ 14  24]
 [ 34 134]]
'''