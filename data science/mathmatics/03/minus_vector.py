import numpy as np
import matplotlib.pyplot as plt

a = np.array([1, 2])
b = np.array([2, 1])
c = a - b

plt.rc("font", size=18) # 그림의 폰트 크기를 18로 고정
gray = {"facecolor": "gray"}
black = {"facecolor": "black"}
red = {"facecolor": "red"}
green = {"facecolor": "green"}
blue = {"facecolor": "blue"}

plt.annotate('', xy=a, xytext=(0, 0), arrowprops=gray)
plt.annotate('', xy=b, xytext=(0, 0), arrowprops=gray)
plt.annotate('', xy=a, xytext=b, arrowprops=black)

plt.plot(0, 0, 'kP', ms=10)
plt.plot(a[0], a[1], 'ro', ms=10)
plt.plot(b[0], b[1], 'ro', ms=10)

plt.text(0.35, 1.15, "$a$")
plt.text(1.15, 0.25, "$b$")
plt.text(1.55, 1.65, "$a-b$")

plt.xticks(np.arange(-2, 5))
plt.yticks(np.arange(-1, 4))

plt.xlim(-0.8, 2.8)
plt.ylim(-0.8, 2.8)

plt.show()