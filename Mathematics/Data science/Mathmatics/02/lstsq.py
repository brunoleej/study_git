# 연립방정식을 풀어주는 명령어
# lstsq() -> 리스트 스퀘어
# lstsq(행렬, 상수벡터)
# Ax = b ==> A(계수행렬),x(미지수벡터),b(상수벡터)
import numpy as np

A = np.array([[1,1,0],[0,1,1],[1,1,1]])
b = np.array([[2],[2],[3]])

x,resid, rank, s = np.linalg.lstsq(A,b)
print(x)
'''
[[1.]
 [1.]
 [1.]]
'''