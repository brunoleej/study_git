import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['figure.figsize'] = (6, 3)
plt.rcParams['font.size'] = 12

a = np.random.standard_normal(1000)
b = np.random.standard_normal(10000)
c = np.random.standard_normal(100000)

plt.hist(a, bins=100, density=True, alpha=0.5, histtype='step', label='n=1000')
plt.hist(b, bins=100, density=True, alpha=0.75, histtype='step', label='n=10000')
plt.hist(c, bins=100, density=True, alpha=1.0, histtype='step', label='n=100000')

plt.legend()
plt.show()