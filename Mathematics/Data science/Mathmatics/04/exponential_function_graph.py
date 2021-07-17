import numpy as np
import matplotlib.pyplot as plt

xx = np.linspace(-2, 2, 100)
yy = np.exp(xx)

plt.title("지수함수")

plt.plot(xx, yy)

plt.axhline(1, c='r', ls="--")
plt.axhline(0, c='r', ls="--")
plt.axvline(0, c='r', ls="--")

plt.xlabel("$x$")
plt.ylabel("$\exp(x)$")

plt.show()
