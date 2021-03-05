import numpy as np
import matplotlib.pyplot as plt

def softmax(x, w): # x는 1차원 배열, w는 가중치 벡터
 e = np.exp(w * x)
 return np.exp(w * x) / e.sum()

x = [2.0, 1.0, 0.5]
y = softmax(x, np.ones(3))

print(y)    # [0.62853172 0.2312239  0.14024438]
print(np.sum(y))    # 1.0
print(softmax(x, 4 * np.ones(3)))   # [0.97962921 0.01794253 0.00242826]