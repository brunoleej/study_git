mu = 0
std = 1
rv = sp.stats.norm(mu, std)

xx = np.linspace(-5, 5, 100)

np.random.seed(0)
x = rv.rvs(20)
print(x)

sns.distplot(x, rug=True, kde=False, fit=sp.stats.norm)
plt.title("랜덤 표본 생성 결과")
plt.xlabel("표본값")
plt.ylabel("$p(x)$")
plt.show()