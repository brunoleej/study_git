import numpy as np

# vector의 dot product
x = np.array([[1],[2],[3]])
y = np.array([[4],[5],[6]])

# dot Product(내적)
print(x.T @ y)  # [[32]]
# print(np.dot(x.T, y))

# 1 dimension array dot product
x1 = np.array([1,2,3])
y1 = np.array([4,5,6])

print(x1.T @ y1)    # 32
# print(np.dot(x1.T @ y1))