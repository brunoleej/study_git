import numpy as np

# 이미 존재하는 array의 shape는 유지하면서 데이터를 '0', '1', '빈 배열'로 반환한다.
k = np.array(range(10)).reshape(2,5)
print(k)
'''
[[0 1 2 3 4]
 [5 6 7 8 9]]
'''

print(np.zeros_like(k))
'''
[[0 0 0 0 0]
 [0 0 0 0 0]]
'''

print(np.ones_like(k))
'''
[[1 1 1 1 1]
 [1 1 1 1 1]]
'''

print(np.empty_like(k))
'''
[1 1 1 1 1]
 [1 1 1 1 1]]
'''
# ??? 원래 빈 반환값을 돌려줘야 하는데 왜이런지 수정 필요