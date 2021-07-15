import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['figure.figsize'] = (6, 3)
plt.rcParams['font.size'] = 12

a = np.random.randn(100000)
b = 2 * np.random.randn(100000) - 1
c = 4 * np.random.randn(100000) + 2

plt.hist(a, bins=100, density=True, alpha=0.5, histtype='step', label='(mean, stddev)=(0, 1)')
plt.hist(b, bins=100, density=True, alpha=0.75, histtype='step', label='(mean, stddev)=(-1, 2)')
plt.hist(c, bins=100, density=True, alpha=1.0, histtype='step', label='(mean, stddev)=(2, 4)')

plt.xlim(-15, 25)
plt.legend()
plt.show()