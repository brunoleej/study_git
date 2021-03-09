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

iris_df.to_scv('../data/csv/iris_sklearn.csv',sep=',')