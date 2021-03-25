import numpy as np
import scipy as sp
import scipy.stats

N = 100
mu_0 = 0
np.random.seed(0)
x = sp.stats.norm(mu_0).rvs(N)

def ztest_1samp(x, sigma2=1, mu=0):
    z = (x.mean() - mu) / np.sqrt(sigma2/len(x))
    return z, 2 * sp.stats.norm().sf(np.abs(z))

print(ztest_1samp(x))   # (0.5980801553448499, 0.5497864508624168)