import numpy as np
import matplotlib.pyplot as plt

def softplus(x):
    return np.log(1 + np.exp(x))
xx = np.linspace(-10, 10, 100)

plt.plot(xx, softplus(xx))
plt.title("소프트플러스함수")
plt.xlabel("$x$")
plt.ylabel("Softplus($x$)")
plt.show()