import numpy as np

a = np.ones(3, dtype=np.int32)
b = np.linspace(0, np.pi, 3)
print(b.dtype.name)     # float64

c = a + b
print(c)                # [1.         2.57079633 4.14159265]
print(c.dtype.name)     # float64

d = np.exp(c * 1j)
print(d)                # [ 0.54030231+0.84147098j -0.84147098+0.54030231j -0.54030231-0.84147098j]
print(d.dtype.name)     # complex128
