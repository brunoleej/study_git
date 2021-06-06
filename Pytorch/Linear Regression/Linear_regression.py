# Linear Regression 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 현재 실습하고 있는 파이썬 코드를 재실행해도 다음에도 같은 결과가 나오도록 랜덤 시드(random seed)를 줍니다.
torch.manual_seed(1)

'''
# Variable 선언
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])


print(x_train)
print(x_train.shape)    # torch.Size([3, 1])

# Weight bias 초기화
# 선형 회귀란 학습 데이터와 가장 잘 맞는 하나의 직선을 찾는 일입니다.
# 그리고 가장 잘 맞는 직선을 정의하는 것은 바로 W와 b입니다.
# 선형 회귀의 목표는 가장 잘 맞는 직선을 정의하는 W와 b의 값을 찾는 것입니다.

# Weight를 0으로 초기화하고, 값 출력 수행
# 가중치 W를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시함.
W = torch.zeros(1, requires_grad=True)
# Weight "W"출력
print(W)    # tensor([0.], requires_grad=True)
# 가중치 W가 0으로 초기화되어있으므로 0이 출력된 것을 확인할 수 있습니다. 
# 위에서 requires_grad=True가 인자로 주어진 것을 확인할 수 있습니다. 이는 이 변수는 학습을 통해 계속 값이 변경되는 변수임을 의미합니다.
b = torch.zeros(1, requires_grad=True)
print(b)    # tensor([0.], requires_grad=True)

# 현재 가중치 W와 b 둘 다 0이므로 현 직선의 방정식은 다음과 같습니다.
# y = 0 x x + 0
# 지금 상태에선 x에 어떤 값이 들어가도 가설은 0을 예측하게 됩니다. 즉, 아직 적절한 W와 b의 값이 아닙니다

# 4. Hypothesis 세우기
# H(x) = Wx + b
hypthesis = x_train * W + b
print(hypthesis)

# 5. 비용 함수 선언하기
# torch.mean으로 mean을 구함
cost = torch.mean((hypthesis - y_train) ** 2)
print(cost) # tensor(18.6667, grad_fn=<MeanBackward0>)

# Gradient Descent 구현
# 학습 대상인 W와 b가 SGD의 입력이 됩니다.
optimizer = optim.SGD([W,b], lr = 0.01)

# optimizer.zero_grad()를 실행하므로서 미분을 통해 얻은 기울기를 0으로 초기화합니다. 
# 기울기를 초기화해야만 새로운 가중치 편향에 대해서 새로운 기울기를 구할 수 있습니다. 
# 그 다음 cost.backward() 함수를 호출하면 가중치 W와 편향 b에 대한 기울기가 계산됩니다. 
# 그 다음 경사 하강법 최적화 함수 opimizer의 .step() 함수를 호출하여 
# 인수로 들어갔던 W와 b에서 리턴되는 변수들의 기울기에 학습률(learining rate) 0.01을 곱하여 빼줌으로서 업데이트합니다.

# Gradient를 0으로 초기화
optimizer.zero_grad()

# 비용 함수를 미분하여 gradient를 계산
cost.backward()

# W와 b를 업데이트
optimizer.step()
'''

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
# 모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 1999 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))