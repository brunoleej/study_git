import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

x = np.random.rand(5)
x = x / x.sum()

plt.subplot(211)
plt.title("0, 1 사이 숫자들의 $\log$ 변환")
plt.bar(range(1, 6), x)
plt.ylim(0, 1)
plt.ylabel("x")
plt.subplot(212)
plt.bar(range(1, 6), np.log(x))
plt.ylabel("log x")
plt.show()
