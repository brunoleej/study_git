import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data
target = iris.target
print(data.shape,target.shape)  # (150, 4) (150,)
# print(iris.keys()) # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
# print(iris.values())
print(iris.target_names)    # ['setosa' 'versicolor' 'virginica']
print(type(data),type(target))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

iris_df = pd.DataFrame(data = iris.data,columns = iris.feature_names)
# print(iris_df)
print(iris_df.shape)   # [150 rows x 4 columns]
print(iris_df.columns)
# Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)'],dtype='object')

# print(iris_df.head(3))
# print(iris_df.tail(4))
print(iris_df.info())   # 결측치 관측 가능
print(iris_df.describe())

iris_df.columns= ['sepal_length','sepal_width','petal_length','petal_width']
print(iris_df.head(3))

# add the y Column
print(iris_df[['sepal_length']])
iris_df['Target'] = target
print(iris_df.head(3))

print(iris_df.shape)
print(iris_df.columns)
print(iris_df.index)    # Index(['sepal length', 'sepal width', 'petal length', 'petal width ','Target'],
print(iris_df.tail)

print(iris_df.info())
print(iris_df.isnull())
print(iris_df.isnull().sum())

print(iris_df['Target'].value_counts())

# Correlation coefficient
print(iris_df.corr())

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(font_scale=1.2)
# sns.heatmap(data=iris_df.corr(),square=True,annot=True,cbar=True)
# plt.show()

# 도수 분포도
plt.figure(figsize = (10,6))

plt.subplot(221)
plt.hist(x = 'sepal_length',data= iris_df)
plt.title('sepal_length')

plt.subplot(222)
plt.hist(x = 'sepal_width',data= iris_df)
plt.title('sepal_width')

plt.subplot(223)
plt.hist(x = 'petal_length',data= iris_df)
plt.title('petal_langth')

plt.subplot(224)
plt.hist(x = 'petal_width',data= iris_df)
plt.title('petal_width')
plt.show()