import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['figure.figsize'] = (6, 3)
plt.rcParams['font.size'] = 12

a = np.random.choice(10, 1000)
b = np.random.choice([0, 1, 2, 4, 8], 1000)

plt.hist(a, bins=100, density=False, alpha=0.75, histtype='step', label='Sample np.arange(5)')
plt.hist(b, bins=100, density=False, alpha=0.75, histtype='step', label='Sample [0, 1, 2, 4, 8]')

plt.ylim(0, 300)
plt.legend()
plt.show()