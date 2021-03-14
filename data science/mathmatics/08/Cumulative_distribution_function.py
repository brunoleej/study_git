import scipy as sp
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

rv = sp.stats.norm(loc=1, scale=2)

xx = np.linspace(-8, 8, 100)
cdf = rv.cdf(xx)
plt.plot(xx, cdf)
plt.title("누적분포함수 ")
plt.xlabel("$x$")
plt.ylabel("$F(x)$")
plt.show()