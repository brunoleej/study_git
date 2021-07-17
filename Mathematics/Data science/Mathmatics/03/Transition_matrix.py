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
a = np.array([2, 2])
g1 = np.array([1, 1]) / np.sqrt(2)
g2 = np.array([-1, 1]) / np.sqrt(2)

plt.annotate('', xy=e1, xytext=(0, 0), arrowprops=green)
plt.annotate('', xy=e2, xytext=(0, 0), arrowprops=green)
plt.annotate('', xy=g1, xytext=(0, 0), arrowprops=red)
plt.annotate('', xy=g2, xytext=(0, 0), arrowprops=red)

plt.text(-0.18, 0.5, "$e_2$", fontdict={"size": 18})
plt.text(0.5, -0.2, "$e_1$", fontdict={"size": 18})
plt.text(0.3, 0.5, "$g_1$", fontdict={"size": 18})
plt.text(-0.45, 0.2, "$g_2$", fontdict={"size": 18})

plt.xticks(np.arange(-2, 4))
plt.yticks(np.arange(-1, 4))

plt.xlim(-1.2, 1.7)
plt.ylim(-0.5, 1.3)

plt.show()
