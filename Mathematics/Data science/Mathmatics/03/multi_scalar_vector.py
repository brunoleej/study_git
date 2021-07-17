import numpy as np
import matplotlib.pyplot as plt

a = np.array([1, 2])
b = 2 * a
c = -a

plt.rc("font", size=18) # 그림의 폰트 크기를 18로 고정
gray = {"facecolor": "gray"}
black = {"facecolor": "black"}
red = {"facecolor": "red"}
green = {"facecolor": "green"}
blue = {"facecolor": "blue"}

plt.annotate('', xy=b, xytext=(0, 0), arrowprops=red)

plt.text(0.8, 3.1, "$2a$")
plt.text(2.2, 3.8, "$(2, 4)$")

plt.annotate('', xy=a, xytext=(0, 0), arrowprops=gray)

plt.text(0.1, 1.3, "$a$")
plt.text(1.1, 1.4, "$(1, 2)$")

plt.plot(c[0], c[1], 'ro', ms=10)

plt.annotate('', xy=c, xytext=(0, 0), arrowprops=blue)

plt.text(-1.3, -0.8, "$-a$")
plt.text(-3, -2.5, "$(-1, -2)$")
plt.plot(0, 0, 'kP', ms=20)

plt.xticks(np.arange(-5, 6))
plt.yticks(np.arange(-5, 6))

plt.xlim(-4.4, 5.4)
plt.ylim(-3.2, 5.2)
plt.show()