# StandardScaler
# 평균이 0에 가깝도록, 분산은 1에 아주 가까운 값으로 변환
import pandas as pd
from sklearn.datasets import load_iris

# 붓꽃 데이터 세트를 로딩하고 DataFrame으로 변환합니다
iris = load_iris()
data = iris.data

iris_df = pd.DataFrame(data=iris.data,columns = iris.feature_names)

print('feature 들의 평균 값')
print(iris_df.mean())
print('\nfeature 들의 분산 값')
print(iris_df.var())

# StandardScaler 적용
from sklearn.preprocessing import StandardScaler

# StandardScaler객체 생성
scaler = StandardScaler()
# StandardScaler로 데이터 세트 변환. fit()과 transform() 호출
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

# transform() 시 스케일 변환된 데이터 세트가 Numpy ndarray로 반환되어 이를 DataFrame으로 변환
iris_df_scaled = pd.DataFrame(data = iris_scaled, columns = iris.feature_names)

print('feature 들의 평균 값')
print(iris_df_scaled.mean())
print('\nfeature 들의 분산 값')
print(iris_df_scaled.var())