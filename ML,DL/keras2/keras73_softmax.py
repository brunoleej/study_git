# softmax 함수

import numpy as np
import matplotlib.pyplot as plt

def softmax(x) :                            # 모두 다 합쳐서 1이 된다
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(1, 5)
y = softmax(x)

ratio = y
labels = y

print(x)
print(y)

plt.pie(ratio, labels=labels, shadow=True, startangle=90)
plt.show()