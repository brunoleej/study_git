import scipy as sp
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

rv = sp.stats.norm(loc=1, scale=2)

xx = np.linspace(-8, 8, 100)
pdf = rv.pdf(xx)
plt.plot(xx, pdf)
plt.title("확률밀도함수 ")
plt.xlabel("$x$")
plt.ylabel("$p(x)$")
plt.show()