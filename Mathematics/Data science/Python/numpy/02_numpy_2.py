import numpy as np

# 1. linspace, logspace Function
# linspace : 설정한 범위에서 선형적으로 분할한 위치의 값을 출력
# logspace : 설정한 범위에서 로그로 분할한 위치의 값을 출력

# linspace
print(np.linspace(0,100,5)) # [  0.  25.  50.  75. 100.]

# logspace
# log10(x1)=2, log10(x2)=3, log10(x3)=4 
print(np.logspace(2,4,3))   # [  100.  1000. 10000.]

# 2. numpy random
# seed: 랜덤값을 설정값
# rand : 균등분포로 난수를 발생
# randn : 정규분포로 난수를 발생
# randint : 균등분포로 정수값을 발생
# suffle : 행렬 데이터를 섞어 줍니다.
# choice : 특정 확률로 데이터를 선택

# seed
np.random.seed(1)
result1 = np.random.randint(10, 100, 10)

np.random.seed(1)
result2 = np.random.randint(10, 100, 10)

np.random.seed(2)
result3 = np.random.randint(10, 100, 10)

print(result1, result2, result3) 
# [47 22 82 19 85 15 89 74 26 11]   => result1
# [47 22 82 19 85 15 89 74 26 11]   => result2
# [50 25 82 32 53 92 85 17 44 59]   => result3

# rand
print(np.random.rand(10))   # [0.20464863 0.61927097 0.29965467 0.26682728 0.62113383 0.52914209 0.13457995 0.51357812 0.18443987 0.78533515]

# randn
print(np.random.randn(10))  # [-0.0191305   1.17500122 -0.74787095  0.00902525 -0.87810789 -0.15643417 0.25657045 -0.98877905 -0.33882197 -0.23618403]

# shuffle
r = np.random.randint(1, 10, (3, 4))
print(r)
'''
[[2 8 9 3]
 [9 8 2 7]
 [9 6 4 1]]
'''
# choice
print(np.random.choice(5, 10, p=[0.1, 0, 0.4, 0.2, 0.3]))   # [4 2 3 2 2 4 3 4 3 0] 
# p => probability

# unique
numbers, counts = np.unique(r, return_counts=True) 
print(numbers)  # [1 2 3 4 6 7 8 9]
print(counts)   # [1 2 1 1 1 1 2 3]

# 3. 행렬 데이터의 결합
# concatenate
na1 = np.random.randint(10, size=(2, 3))    # 2 Row, 3 Column
na2 = np.random.randint(10, size=(3, 2))    # 3 Row, 2 Column
na3 = np.random.randint(10, size=(3, 3))    # 3 Row, 3 Column

# 세로 결합
print(np.concatenate((na1, na3)))
'''
[[1 4 8]
 [1 6 9]
 [4 5 8]
 [3 0 0]
 [5 7 5]]
'''

# 가로 결합
print(np.concatenate((na2, na3), axis=1))
'''
[[5 1 4 5 8]
 [2 4 3 0 0]
 [7 6 5 7 5]]
'''

# c_, r_
print(np.c_[np.array([1,2,3]), np.array([4,5,6])])
'''
[[1 4]
 [2 5]
 [3 6]]
'''
print(np.r_[np.array([1,2,3]), np.array([4,5,6])])  # [1 2 3 4 5 6]
