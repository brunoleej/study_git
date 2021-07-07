import numpy as np

a = np.array([0, np.pi / 2, np.pi])
print(np.sin(a))
print(np.cos(a))

b = np.arange(3)
print(b)
print(np.exp(b))
print(np.sqrt(b))

c1 = np.array([2, -1, 4], dtype=float)
print(c1)
print(c1.dtype)
c2 = np.array([2., -1, 4.])
print(c2)
print(c2.dtype)

print(np.add(b, c1))