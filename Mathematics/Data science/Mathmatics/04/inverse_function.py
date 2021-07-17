import numpy as np
import matplotlib.pyplot as plt

def f1(x):
     return x ** 2

def f1inv(x):
 return np.sqrt(x)

x = np.linspace(0, 3, 300)

plt.plot(x, f1(x), "r-", label="함수 $f(x) = x^2$")
plt.plot(x, f1inv(x), "b-.", label="역함수 $f^{-1}(x) = \sqrt{x}$")
plt.plot(x, x, "g--")

plt.axis("equal")

plt.xlim(0, 2)
plt.ylim(0, 2)

plt.legend()
plt.title("역함수의 그래프")

plt.show()
