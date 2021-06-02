# sigmoid 함수

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x) :
    return 1 / (1 + np.exp(-x)) # np.exp : 자연상수 e 인 지수함수로 만들어준다.

x = np.arange(-5, 5, 0.1)   # -5부터 5까지 0.1 간격으로
y = sigmoid(x)              # x 에 대한 sigmoid (결과 값이 0과 1 사이로 수렴된다.)

print(x)
print(y)

# 시각화
plt.plot(x, y)
plt.grid()
plt.show()