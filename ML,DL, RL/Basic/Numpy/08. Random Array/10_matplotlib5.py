import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['figure.figsize'] = (6, 3)
plt.rcParams['font.size'] = 12

a = np.random.normal(0, 1, 500)
b = np.random.normal(1.5, 1.5, 5000)
c = np.random.normal(3.0, 2.0, 50000)

plt.hist(a, bins=100, density=True, alpha=0.75, histtype='step', label=r'N(0, $1^2$)')
plt.hist(b, bins=100, density=True, alpha=0.75, histtype='step', label=r'N(1.5, $1.5^2$)')
plt.hist(c, bins=100, density=True, alpha=0.75, histtype='step', label=r'N(3.0, $3.0^2$)')

plt.legend()
plt.show()