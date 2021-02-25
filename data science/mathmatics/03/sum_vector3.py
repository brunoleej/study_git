import numpy as np
import matplotlib.pyplot as plt

a = np.array([1, 2])
b = np.array([2, 1])
c = a + b

plt.rc("font", size=18) # 그림의 폰트 크기를 18로 고정
gray = {"facecolor": "gray"}
black = {"facecolor": "black"}
red = {"facecolor": "red"}
green = {"facecolor": "green"}
blue = {"facecolor": "blue"}

plt.annotate('', xy=b, xytext=(0, 0), arrowprops=gray)
plt.annotate('', xy=c, xytext=b, arrowprops=gray)
plt.annotate('', xy=c, xytext=(0, 0), arrowprops=black)

plt.plot(0, 0, 'kP', ms=10)
plt.plot(b[0], b[1], 'ro', ms=10)
plt.plot(c[0], c[1], 'ro', ms=10)

plt.text(2.45, 1.55, "$a$")
plt.text(1.25, 0.25, "$b$")
plt.text(1.25, 1.45, "$c$")

plt.xticks(np.arange(-2, 5))
plt.yticks(np.arange(-1, 4))

plt.xlim(-1.4, 4.4)
plt.ylim(-0.6, 3.8)

plt.show()
