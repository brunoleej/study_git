import numpy as np

def f(x, y):
  return x + y

b = np.fromfunction(f, (5, 4), dtype=int)

for row in b:
  print(row)