import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 5, 100)
y = np.exp(x)

plt.plot(x, y)
# plt.yscale('linear')
plt.yscale('log')

plt.show()