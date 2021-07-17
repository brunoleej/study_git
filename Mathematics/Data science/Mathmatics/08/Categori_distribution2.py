import scipy.stats as sp
import matplotlib.pyplot as plt

mu = [0.1, 0.1, 0.1, 0.1, 0.3, 0.3]
rv = sp.stats.multinomial(1, mu)

np.random.seed(1)
X = rv.rvs(100)
print(X[:5])

y = X.sum(axis=0) / float(len(X))
plt.bar(np.arange(1, 7), y)
plt.title("카테고리분포의 시뮬레이션 결과")
plt.xlabel("표본값")
plt.ylabel("비율")
plt.show()

df = pd.DataFrame({"이론": rv.pmf(xx_ohe.values), "시뮬레이션": y},index=np.arange(1, 7)).stack()
df = df.reset_index()
df.columns = ["표본값", "유형", "비율"]
df.pivot("표본값", "유형", "비율")
df