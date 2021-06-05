# 비용함수를 직접 정의해서 선형회귀 모델 구현
# nn.Linear()함수로, 평균제곱오차 nn.functional.mse_loss()라는 함수로 구현되어져 있음.
# model = nn.Linear(input_dim, output_dim)
# cost = F.mse_loss(prediction, y_train)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Data
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

# Model => Model 선언 및 초기화. Simple Lienar Regression이므로 input_dim = 1, output_dim = 1
model = nn.Linear(1,1)
# 하나의 입력 x에 대해서 하나의 출력 y를 가짐

# model에는 Weight W와 Bias b가 저장되어져 있음. => model.parameter()로 불러올 수 있음
print(list(model.parameters()))
'''
[Parameter containing:
tensor([[0.5153]], requires_grad=True), Parameter containing:
tensor([-0.4414], requires_grad=True)]
'''
# => 2개의 값이 출력되는데 첫 번째 값이 W이고 두 번째 값이 b에 해당됨.
# 두 값 모두 현재는 초기화 되어져 있음. 두값 모두 학습의 대상이므로 requires_grad = True가 되어져 있는 것을 볼 수 있음

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# 전체 훈련 데이터에 대해 GD를 2,000회 반복
EPOCHS = 2000

for epoch in range(EPOCHS + 1):
    # H(x) computation
    prediction = model(x_train)

    # cost
    cost = torch.nn.functional.mse_loss(prediction, y_train) # <== Pytorch에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선
    optimizer.zero_grad()   # gradient를 0으로 초기화
    cost.backward()         # 비용 함수를 미분하여 gradient를 계산
    optimizer.step()        # W와 b를 업데이트

    if epoch % 100 == 0:
         print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, EPOCHS, cost.item()))

'''
Epoch    0/2000 Cost: 13.103541
Epoch  100/2000 Cost: 0.002791
Epoch  200/2000 Cost: 0.001724
Epoch  300/2000 Cost: 0.001066
Epoch  400/2000 Cost: 0.000658
Epoch  500/2000 Cost: 0.000407
Epoch  600/2000 Cost: 0.000251
Epoch  700/2000 Cost: 0.000155
Epoch  800/2000 Cost: 0.000096
Epoch  900/2000 Cost: 0.000059
Epoch 1000/2000 Cost: 0.000037
Epoch 1100/2000 Cost: 0.000023
Epoch 1200/2000 Cost: 0.000014
Epoch 1300/2000 Cost: 0.000009
Epoch 1400/2000 Cost: 0.000005
Epoch 1500/2000 Cost: 0.000003
Epoch 1600/2000 Cost: 0.000002
Epoch 1700/2000 Cost: 0.000001
Epoch 1800/2000 Cost: 0.000001
Epoch 1900/2000 Cost: 0.000000
Epoch 2000/2000 Cost: 0.000000
'''

# cost 값이 매우 작은데 W 와 b의 값도 최적화가 되었는지 확인
# 임의의 입력 4를 선언
new_var = torch.FloatTensor([[4.0]])

# 입력한 값 4에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) # forward 연산

# y = 2x이므로 입력이 4라면 y가 8에 가까운 값이 나와야 제대로 학습이 된 것
print('훈련 후 입력이 4일때의 예측값 : {}'.format(pred_y))  # 훈련 후 입력이 4일때의 예측값 : tensor([[7.9989]], grad_fn=<AddmmBackward>)

# 학습 후의 W와 b값을 출력
print(list(model.parameters()))
'''
[Parameter containing:
tensor([[1.9994]], requires_grad=True), Parameter containing:
tensor([0.0014], requires_grad=True)]
'''

# H(x) 식에 입력 x로부터 예측된 y를 얻는 것을 forward 연산이라고 합니다.
# 학습 전, prediction = model(x_train)은 x_train으로부터 예측값을 리턴하므로 forward 연산입니다.
# 학습 후, pred_y = model(new_var)는 임의의 값 new_var로부터 예측값을 리턴하므로 forward 연산입니다.
# 학습 과정에서 비용 함수를 미분하여 기울기를 구하는 것을 backward 연산이라고 합니다.
# cost.backward()는 비용 함수로부터 기울기를 구하라는 의미이며 backward 연산입니다.
