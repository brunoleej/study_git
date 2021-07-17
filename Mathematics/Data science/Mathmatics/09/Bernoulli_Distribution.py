import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

mu0 = 0.65
a, b = 1, 1
print("초기 추정: 모드 = 모름")
xx = np.linspace(0, 1, 1000)
plt.plot(xx, sp.stats.beta(a, b).pdf(xx), ls=":", label="초기 추정")
np.random.seed(0)

for i in range(3):
    x = sp.stats.bernoulli(mu0).rvs(50)
    N0, N1 = np.bincount(x, minlength=2)
    a, b = a + N1, b + N0
    plt.plot(xx, sp.stats.beta(a, b).pdf(xx), ls="-.", label="{}차 추정".format(i))
    print("{}차 추정: 모드 = {:4.2f}".format(i, (a - 1)/(a + b - 2)))

plt.vlines(x=0.65, ymin=0, ymax=12)
plt.ylim(0, 12)
plt.legend()
plt.title("베르누이분포의 모수를 베이즈 추정법으로 추정한 결과")
plt.show()