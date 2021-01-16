import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data

iris_df = pd.DataFrame(data = data,columns = iris.feature_names)

from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler객체 생성
scaler = MinMaxScaler()
# MinMaxScaler로 데이터 세트 변환. fit()과 transform() 호출
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

# transform() 시 스케일 변환된 데이터 세트가 Numpy ndarray로 변환 되 이를 DataFrame으로 변환
iris_df_scaled = pd.DataFrame(data = iris_scaled, columns = iris.feature_names)
print('feature 들의 최솟값')
print(iris_df_scaled.min())
print('\nfeature 들의 최댓값')
print(iris_df_scaled.max())

'''
feature 들의 최솟값
sepal length (cm)    0.0
sepal width (cm)     0.0
petal length (cm)    0.0
petal width (cm)     0.0
dtype: float64

feature 들의 최댓값
sepal length (cm)    1.0
sepal width (cm)     1.0
petal length (cm)    1.0
petal width (cm)     1.0
dtype: float64
'''

# Problem
# 학습 데이터와 테스트 데이터의 스케일링 변환 시 유의점
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 학습 데이터는 0부터 10까지, 테스트 데이터는 0부터 5까지의 값을 가지는 데이터 세트로 생성
# Scaler 클래스의 fit(), transform()은 2차원 이상의 데이터만 가능하므로 reshape(-1,1)을 통해 차원 변경
train_array = np.arange(0,11).reshape(-1,1)
test_array = np.arange(0,6).reshape(-1,1)

# train_array부터 MinMaxScaler를 이용해 변환
# MinMaxScaler 객체에 별도의 feature_range 파라미터 값을 지정하지 않으면 0~1 값으로 변환
scaler = MinMaxScaler()

# fit()하게 되면 train_array 데이터의 최솟값이 0, 최댓값이 10으로 설정
scaler.fit(train_array)

# 1/10 scale로 train_array 데이터 변환함. 원본 10 -> 1로 변환됨
train_scaled = scaler.transform(train_array)

print('원본 train_array 데이터: ',np.round(train_array.reshape(-1),2))
print('Scale된 train_array 데이터: ',np.round(train_scaled.reshape(-1),2))

'''
원본 train_array 데이터:  [ 0  1  2  3  4  5  6  7  8  9 10]
Scale된 train_array 데이터:  [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
'''

# test_array를 MinMaxScaler를 이용해 변환
# MinMaxScaler에 test_array를 fit()하게 되면 원본 데이터의 최솟값이 0, 최댓값이 5로 설정됨
scaler.fit(test_array)

# 1/5 scale로 test_array 데이터 변환함, 원본 5 -> 1로 변환.
test_scaled = scaler.transform(test_array)

# test_array의 scale 변환 출력
print('원본 test_array 데이터: ',np.round(test_array.reshape(-1),2))
print('Scale된 test_array 데이터: ',np.round(test_scaled.reshape(-1),2))

'''
원본 test_array 데이터:  [0 1 2 3 4 5]
Scale된 test_array 데이터:  [0.  0.2 0.4 0.6 0.8 1. ]
'''