import numpy as np

# ravel : high-dimensional array to translation 1 dimension
# C : row
# F column
x = np.arange(15).reshape(3,5)
print(x)

print(np.ravel(x, order='C'))   # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
print(np.ravel(x, order='F'))   # [ 0  5 10  1  6 11  2  7 12  3  8 13  4  9 14]

tmp = x.ravel()
print(tmp)                      # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]

tmp[0] = 100
print(tmp)                      # [100   1   2   3   4   5   6   7   8   9  10  11  12  13  14]
print(x)

# flatten : high-dimensional array translation to 1 dimensional array
y = np.arange(15).reshape(3,5)
print(y)

t2 = y.flatten(order = 'F')
print(t2)           # [ 0  5 10  1  6 11  2  7 12  3  8 13  4  9 14]

t2[0] = 100
print(t2)           # [100   5  10   1   6  11   2   7  12   3   8  13   4   9  14]
print(y)

x1 = np.arange(30).reshape(2,3,5)
print(x1)

print(x1.ravel())   # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29]

# reshape
x2 = np.arange(36)  
print(x2)           # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35]
print(x2.shape)     # (36,)
print(x2.ndim)      # 1

y2 = x2.reshape(6,6)
print(y2.shape)     # (6, 6)
print(y2.ndim)      # 2

k = x2.reshape(3,3,-1)
print(k.shape)      # (3, 3, 4)
print(k.ndim)       # 3