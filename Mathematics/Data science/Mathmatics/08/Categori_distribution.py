import scipy.stats as sp
import matplotlib.pyplot as plt

mu = [0.1, 0.1, 0.1, 0.1, 0.3, 0.3]
rv = sp.stats.multinomial(1, mu)

xx = np.arange(1, 7)
xx_ohe = pd.get_dummies(xx)
plt.bar(xx, rv.pmf(xx_ohe.values))
plt.ylabel("P(x)")
plt.xlabel("표본값")
plt.title("카테고리분포의 확률질량함수")
plt.show()

np.random.seed(1)
X = rv.rvs(100)
print(X[:5])

