# Numpy를 사용하여 신경망을 구성

import numpy as np

# N은 배치 크기이며, D_in은 입력의 차원입니다.
# H는 은닉층의 차원이며, D_out은 출력 차원입니다.
N, D_in, H, D_out = 64, 1000, 100, 10

# 무작위의 입력과 출력 데이터를 생성합니다.
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# 무작위로 가중치를 초기화합니다.
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6

for t in range(500):
    # 순전파 단계 : 예측값 y를 계산합니다.
    h = x.dot(w1)
    h_relu = np.maximun(h, 0)
    y_pred = h_relu.dot(w2)

    # 손실(loss)을 계산하고 출력합니다.
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # 손실에 따른 w1, w2의 변화도를 계산하고 역전파합니다.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # 가중치를 갱신
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2