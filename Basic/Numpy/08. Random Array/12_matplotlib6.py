import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['figure.figsize'] = (6, 3)
plt.rcParams['font.size'] = 12

a = np.random.random_sample(100000)
b = 1.5 * np.random.random_sample(100000) - 0.75
c = 2 * np.random.random_sample(100000) - 1

plt.hist(a, bins=100, density=True, alpha=0.75, histtype='step', label='[0, 1)')
plt.hist(b, bins=100, density=True, alpha=0.75, histtype='step', label='[-0.75, 0.75)')
plt.hist(c, bins=100, density=True, alpha=0.75, histtype='step', label='[-1, 1)')

plt.ylim(0.0, 1.2)
plt.legend()
plt.show()