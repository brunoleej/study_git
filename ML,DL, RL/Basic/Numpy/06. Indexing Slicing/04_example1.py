import numpy as np

def f(x, y):
  return x + y

b = np.fromfunction(f, (5, 4), dtype=int)

print(b)
print(b[2, 3])      # element of third row, fourth column

print(b[0:5, 1])    # second column
print(b[:, 1])      # second column
print(b[1:3, :])    # second~third row