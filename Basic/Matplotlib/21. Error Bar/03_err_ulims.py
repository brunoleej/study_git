import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 5)
y = x**2
yerr = np.linspace(0.1, 0.4, 4)

plt.errorbar(x, y + 4, yerr=yerr)
plt.errorbar(x, y + 2, yerr=yerr, uplims=True, lolims=True)

upperlimits = [True, False, True, False]
lowerlimits = [False, False, True, True]
plt.errorbar(x, y, yerr=yerr, uplims=upperlimits, lolims=lowerlimits)
plt.show()