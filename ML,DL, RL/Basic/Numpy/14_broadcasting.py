import numpy as np

# Same Shape operation 
x = np.arange(15).reshape(3,5)
y = np.random.rand(15).reshape(3,5)
print(x)
print(y)
print(x * y)
print(x % 2 == 0)

# different shape operation
a = np.arange(12).reshape(4, 3)
b = np.arange(100, 103)
c = np.arange(1000, 1004)
d = b.reshape(1, 3)

print(a.shape)  # (4, 3)
print(b.shape)  # (3,)
print(c.shape)  # (4,)
print(d.shape)  # (1, 3)
print(d)        # [[100 101 102]]
print(a + b)
print(a + c)
print(a + d)