import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# W값의 변화에 따른 경사도의 변화
x1 = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5 * x1)
y2 = sigmoid(x1)
y3 = sigmoid(2 * x1)

plt.plot(x1, y1, 'r', linestyle = '--') # W 값이 0.5일때
plt.plot(x1, y2, 'g')   # W 값이 1일때
plt.plot(x1, y3, 'b', linestyle = '--') # W 값이 2일때
plt.plot([0,0], [1.0, 0.0], ':')    # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

# Linear Regression에서는 W가 직선의 기울기를 의미했지만, 여기서는 그래프의 경사도를 결정
# W의 값이 커지면 경사가 커지고 W값이 작아지면 경사가 작아짐

