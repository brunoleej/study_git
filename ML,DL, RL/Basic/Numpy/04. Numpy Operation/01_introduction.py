import numpy as np

a = np.array([20, 30, 40, 50])
b = np.arange(4)

c = a - b
print(c)                # [20 29 38 47]

print(b ** 2)           # [0 1 4 9]
print(10 * np.sin(a))   # [ 9.12945251 -9.88031624  7.4511316  -2.62374854]
print(a < 35)           # [ True  True False False]