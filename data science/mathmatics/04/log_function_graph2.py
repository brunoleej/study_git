import numpy as np
import matplotlib.pyplot as plt

def ff(x):
    return x**3 - 12*x + 20 * np.sin(x) + 7

xx = np.linspace(-4, 4, 300)
yy = ff(xx)

plt.subplot(211)
plt.plot(xx, yy)

plt.axhline(1, c='r', ls="--")
plt.yticks([0, 1, 5, 10])

plt.ylim(-2, 15)
plt.title("$f(x)$")
plt.subplot(212)
plt.plot(xx, np.log(yy))
plt.axhline(0, c='r', ls="--")
plt.title("$log f(x)$")

plt.tight_layout()
plt.show()