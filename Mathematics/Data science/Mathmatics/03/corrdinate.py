import numpy as np
import matplotlib.pylab as plt

gray = {"facecolor": "gray"}
black = {"facecolor": "black"}
red = {"facecolor": "red"}
green = {"facecolor": "green"}
blue = {"facecolor": "blue"}
lightgreen = {"facecolor": "lightgreen"}

e1 = np.array([1, 0])
e2 = np.array([0, 1])
x = np.array([2, 2])

plt.annotate('', xy=2 * e1, xytext=(0, 0), arrowprops=gray)
plt.annotate('', xy=2 * e2, xytext=(0, 0), arrowprops=gray)
plt.annotate('', xy=e1, xytext=(0, 0), arrowprops=green)
plt.annotate('', xy=e2, xytext=(0, 0), arrowprops=green)
plt.annotate('', xy=x, xytext=(0, 0), arrowprops=gray)

plt.plot(0, 0, 'ro', ms=10)
plt.plot(x[0], x[1], 'ro', ms=10)

plt.text(1.05, 1.35, "$x$", fontdict={"size": 18})
plt.text(-0.3, 0.5, "$e_2$", fontdict={"size": 18})
plt.text(0.5, -0.2, "$e_1$", fontdict={"size": 18})

plt.xticks(np.arange(-2, 4))
plt.yticks(np.arange(-1, 4))

plt.xlim(-1.5, 3.5)
plt.ylim(-0.5, 3)

plt.show()