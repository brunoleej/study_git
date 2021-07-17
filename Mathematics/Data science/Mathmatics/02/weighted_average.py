import numpy as np

x = np.arange(10)
N = len(x)

# numpy로 계산
# print(np.ones(N) @ x / N)   # 4.5

# 위의 수식보다는 mean() Method를 사용하는 것이 편하다.
print(x.mean()) # 4.5