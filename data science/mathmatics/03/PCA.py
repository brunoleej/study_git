import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

N = 10 # 앞의 10송이만 선택
X = iris.data[:N, :2] # 꽃받침 길이와 꽃받침 폭만 선택

plt.plot(X.T, 'o:')

plt.xticks(range(4), ["꽃받침 길이", "꽃받침 폭"])

plt.xlim(-0.5, 2)
plt.ylim(2.5, 6)

plt.title("붓꽃 크기 특성")

plt.legend(["표본 {}".format(i + 1) for i in range(N)])

plt.show()