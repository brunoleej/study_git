import numpy as np

x = np.arange(15).reshape(3,5)
y = np.random.rand(15).reshape(3,5)
print(x)
print(y)

# operator function
# add, substract, multiply, divide
print(np.add(x,y))
print(np.divide(x,y))
print(x + y)
print(x / y)

# statistics function
print(y)
print(np.mean(y))   # print(y.mean())
print(np.max(y))
print(np.argmax(y)) # 1
print(np.var(y), np.median(y), np.std(y))

# summation function
print(np.sum(y, axis = None))
print(np.cumsum(y))

z = np.random.randn(10)
print(z)                # [-0.5070304   1.36381246 -0.25131361  0.63909562  0.67120275 -0.0474122 -1.22961458  0.59172759  0.65431842  0.16166565]
print(z > 0)            # [False  True False  True  True False False  True  True  True]
print(np.all(z != 0))   # True


# where
z2 = np.random.randn(10)
print(z2)   # [-0.86653225  0.48122594 -0.01735896  0.42016904  0.82295035  1.00075112 1.01752545 -1.29299492  1.41373717 -1.44172731]
print(np.where(z > 0, z, 0))