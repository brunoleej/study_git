import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# W : 1, b : 0 graph
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, 'g')
plt.plot([0,0],[1.0, 0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()


