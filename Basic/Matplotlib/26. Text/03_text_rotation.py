import matplotlib.pyplot as plt
import numpy as np

a = 2.0 * np.random.randn(10000) + 1.0
b = np.random.standard_normal(10000)
c = 20.0 * np.random.rand(5000) - 10.0

font1 = {'family': 'serif',
      'color':  'darkred',
      'weight': 'normal',
      'size': 16}

font2 = {'family': 'Times New Roman',
      'color':  'blue',
      'weight': 'bold',
      'size': 12,
      'alpha': 0.7}

font3 = {'family': 'Arial',
      'color':  'forestgreen',
      'style': 'italic',
      'size': 14}

plt.hist(a, bins=100, density=True, alpha=0.7, histtype='step')
plt.text(-3.0, 0.15, 'np.random.randn()', fontdict=font1, rotation=85)
plt.hist(b, bins=50, density=True, alpha=0.5, histtype='stepfilled')
plt.text(2.0, 0.0, 'np.random.standard_normal()', fontdict=font2, rotation=-60)
plt.hist(c, bins=100, density=True, alpha=0.9, histtype='step')
plt.text(-10.0, 0.08, 'np.random.rand()', fontdict=font3)
plt.show()