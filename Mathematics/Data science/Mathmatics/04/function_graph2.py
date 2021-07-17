import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 3 * x**2 + x

x = np.linspace(-1, 3, 9)
print(x)    # [-1.  -0.5  0.   0.5  1.   1.5  2.   2.5  3. ]

y = f(x)
print(y)    # [-5.    -1.375  0.    -0.125 -1.    -1.875 -2.    -0.625  3.   ]

x = np.linspace(-1, 3, 400)
y = f(x)
plt.plot(x, y)

plt.xlim(-2, 4)

plt.title("함수 $f(x) = x^3 - 3x^2 + x$의 그래프")

plt.xlabel("x")
plt.ylabel("y")

plt.xticks(np.arange(-1, 4))
plt.yticks(np.arange(-5, 4))

plt.show()
