import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x + 0.5)
y2 = sigmoid(x + 1)
y3 = sigmoid(x + 1.5)

plt.plot(x, y1, 'r', linestyle = '--')    # x + 0.5
plt.plot(x, y2, 'g')                      # x + 1
plt.plot(x, y3, 'b', linestyle = '--')    # x + 1.5
plt.plot([0,0],[1.0, 0.0], '--')
plt.title('Sigmoid Function')
plt.show()

# sigmoid 함수는 입력값이 한없이 커지면 1에 수렴하고, 입력값이 한없이 작아지면 0에 수렴
# sigmoid 함수는 0과 1사이의 값을 가지는데 이 특성을 이용해서 분류작업을 수행할 수 있음
# 예를 들어, 임계값이 0.5 이상이면 1(True), 0.5 이하면 0(False)으로 판단할 수 있음

