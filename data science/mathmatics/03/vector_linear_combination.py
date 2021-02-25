import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([1, 2])
x2 = np.array([2, 1])
x3 = 0.5 * x1 + x2

plt.rc("font", size=18) # 그림의 폰트 크기를 18로 고정
gray = {"facecolor": "gray"}
black = {"facecolor": "black"}
red = {"facecolor": "red"}
green = {"facecolor": "green"}
blue = {"facecolor": "blue"}

plt.annotate('', xy=0.5*x1, xytext=(0, 0), arrowprops=gray)
plt.annotate('', xy=x2, xytext=(0, 0), arrowprops=gray)
plt.annotate('', xy=x3, xytext=(0, 0), arrowprops=black)

plt.plot(0, 0, 'kP', ms=10)
plt.plot(x1[0], x1[1], 'ro', ms=10)
plt.plot(x2[0], x2[1], 'ro', ms=10)
plt.plot(x3[0], x3[1], 'ro', ms=10)
plt.plot([x1[0], 0], [x1[1], 0], 'k--')

plt.text(0.6, 2.0, "$x_1$")
plt.text(-0.5, 0.5, "$0.5x_1$")
plt.text(1.15, 0.25, "$x_2$")
plt.text(2.5, 1.6, "$0.5x_1 + x_2$")

plt.xticks(np.arange(-2, 5))
plt.yticks(np.arange(-1, 4))

plt.xlim(-1.4, 4.4)
plt.ylim(-0.6, 3.8)

plt.show()