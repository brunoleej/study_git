import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['figure.figsize'] = (6, 3)
plt.rcParams['font.size'] = 12

a = np.random.randint(0, 10, 1000)
b = np.random.randint(10, 20, 1000)
c = np.random.randint(0, 20, 1000)

plt.hist(a, bins=100, density=False, alpha=0.5, histtype='step', label='0<=randint<10')
plt.hist(b, bins=100, density=False, alpha=0.75, histtype='step', label='10<=randint<20')
plt.hist(c, bins=100, density=False, alpha=1.0, histtype='step', label='0<=randint<20')

plt.ylim(0, 150)
plt.legend()
plt.show()