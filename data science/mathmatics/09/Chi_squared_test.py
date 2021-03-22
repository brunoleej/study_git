import numpy as np
import scipy as sp
import scipy.stats

N = 10
K = 4
mu_0 = np.ones(K)/K
np.random.seed(0)
x = np.random.choice(K, N, p=mu_0)
n = np.bincount(x, minlength=K)

print(n)    # [0 3 5 2]
print(sp.stats.chisquare(n))    # Power_divergenceResult(statistic=5.199999999999999, pvalue=0.157724450396663)

obs = np.array([[5, 15], [10, 20]])
print(sp.stats.chi2_contingency(obs))
'''
(0.0992063492063492, 0.7527841326498471, 1, array([[ 6., 14.],
       [ 9., 21.]]))
'''