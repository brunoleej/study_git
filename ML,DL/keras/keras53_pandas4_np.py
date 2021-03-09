import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data
target = iris.target

iris_df = pd.DataFrame(data = iris.data,columns = iris.feature_names)

iris_df.columns= ['sepal_length','sepal_width','petal_length','petal_width']

# add the y Column
print(iris_df[['sepal_length']])
iris_df['Target'] = target
print(iris_df.shape)
print(iris_df.info())

iris_np = iris_df.values
print(iris_np)

np.save('../data/npy/iris_sklearn.npy',arr=iris_np)

# 과제
# 판다스의 loc iloc에 대해 정리

# iloc
print(iris_df.iloc[2:3])

# loc
print(iris_df.head(3))
# print(iris_df.loc[0])