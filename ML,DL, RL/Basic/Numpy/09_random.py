import numpy as np

# rand
# 0 ~ 1 사이의 분포로 랜덤한 ndarray 생성
x1 = np.random.rand(4,5,3)
print(x1)


# randn
# n : normal distribution
# normal distribution으로 sampling된 ndarray 생성
x2 = np.random.randn(5)
print(x2)

x3 = np.random.randn(3,4,2)
print(x3)

# randint
# Random한 값을 동일하게 다시 생성하고자 할 때 사용
np.random.seed(23)  # 랜덤값을 일정하게 유지하고자 할 떄 사용
x4 = np.random.randint(3,4)

# choice
# 주어진 1차원 ndarray로 부터 랜덤으로 샘플링
# 번구다 부너빔 뎐누, np.Enfw(해당 숫자)로 간주
x5 = np.random.choice(100, size = (3,4))
print(x5)

x6 = np.array([1, 2, 3, 1.5, 2.6, 4.9])
print(np.random.choice(x6, size = (2,2), replace=False))

# probability distribution에 따른 ndarray 생성
x7 = np.random.uniform(1.0, 3.0, size = (4,5))
print(x7)

x8 = np.random.normal(size=(3,4))
print(x8)

x9 = np.random.randn(3,4)
print(x9)