import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
# print(iris)
print(iris.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

data = iris.data
target = iris.target
# data = iris['data']
# target = iris['target']
# print(data)
# print(target)
# print(iris.fra)
print(iris.frame)
print(iris.target_names)
print(iris['DESCR'])
print(iris['feature_names'])
print(iris.filename)

print(type(data),type(target))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# np.save('./data/iris_data.npy',arr = data)
# np.save('./data/iris_target.npy',arr = target)