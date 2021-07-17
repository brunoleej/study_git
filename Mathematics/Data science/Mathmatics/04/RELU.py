import numpy as np
import matplotlib.pyplot as plt

xx = np.linspace(-10, 10, 100)

plt.plot(xx, np.maximum(xx, 0))

plt.title("max(x,0) 또는 ReLU")

plt.xlabel("$x$")
plt.ylabel("$ReLU(x)$")

plt.show()