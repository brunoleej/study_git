# package
import numpy as np
import matplotlib.pyplot as plt

# Data
# Scalar
# vector
# matrix
# tensor
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.data[0,:])  # 첫 번째 꽃의 Data

print(iris.feature_names)
# ['sepal length (cm)',
#  'sepal width (cm)',
#  'petal length (cm)',
#  'petal width (cm)']

# from sklearn.datasets import load_digits
# data = load_digits()

# X = data.data
