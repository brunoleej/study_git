import numpy as np
# np.array와의 차이점
# 이미 ndarray의 데이터 형태(data type)이 설정되어 있다면, 데이터 형태가 다를 경우에만 복사(copy)가 된다.

# 이미 존재하는 배열에 대해서는 복사하지 않는다.
a = np.array([1,2,3])
print(np.asarray(a) is a)   # True

# 만약 dtype이 설정되어 있다면, 데이터 형태가 다를 경우에만 복사한다.
b = np.array([1,2,3],dtype = np.float32)

# np.array는 데이터의 형태 일치 여부에 상관없이 복사함.
print(np.array(b, dtype = np.float32) is b) # False

# np.asarray
# 데이터 형태가 같을 떄 복사하지 않는다.
print(np.asarray(b, dtype = np.float32) is b)   # True

# 데이터 형태가 다를때 복사한다.
print(np.asarray(b, dtype = np.float64) is b)   # False
