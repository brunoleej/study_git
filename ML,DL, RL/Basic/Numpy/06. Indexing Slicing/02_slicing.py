import numpy as np

a = np.arange(10) ** 2

# Slicing
print(a[2:5])
print(a[:4:2])  # equal to a[0:4:2]
print(a[4::2])  # eqaul to a[4:len(a):3]
print(a[::-1])  # reversed array