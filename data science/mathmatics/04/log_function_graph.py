import numpy as np
import matplotlib.pyplot as plt

xx = np.linspace(0.01, 8, 100)
yy = np.log(xx)

plt.title("로그함수")
plt.plot(xx, yy)

plt.axhline(0, c='r', ls="--")
plt.axvline(0, c='r', ls="--")
plt.axvline(1, c='r', ls="--")

plt.xlabel("$x$")
plt.ylabel("$\log(x)$")

plt.show()
