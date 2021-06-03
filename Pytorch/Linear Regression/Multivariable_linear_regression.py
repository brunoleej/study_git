# x가 1개인 Linear Regression을 단순 선형 회귀(Simple Linear Regression)이라고 함.
# 다수의 x로부터 y를 예측하는 다중 선형 회귀(Multivariable Linear Regression)이라고 함.

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

# 다시 돌려서 같은 값이 나오게 하기 위해 Random_seed값 고정
torch.manual_seed(1)

# train data
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# Weight "w", Bias "B" 선언. Weight "w"도 3개 선언해주어야 함
# weight "w"와 Bias "b" 초기화
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Hypothesis, Cost function, Optimizer를 선헌 후에 Gradient Descent 1,000회 반복
# optimizer 설정
optimizer = optim.SGD([w1, w2, w3, b], lr = 1e-5)

EPOCHS = 1000
for epochs in range(EPOCHS + 1):
    # H(x) 계산
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
    # cost
    cost = torch.mean((hypothesis - y_train) **2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epochs % 1000 == 0:
         print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epochs, EPOCHS, w1.item(), w2.item(), w3.item(), b.item(), cost.item()))

# result
# Epoch    0/1000 w1: 0.294 w2: 0.294 w3: 0.297 b: 0.003 Cost: 29661.800781
# Epoch 1000/1000 w1: 0.718 w2: 0.613 w3: 0.680 b: 0.009 Cost: 1.079378


# 위의 경우 Hypothesis를 선언하는 부분인 hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3  + b에서도 x_train의 개수만큼 w와 곱해주도록 작성하였음

# 위의 코드를 개선할 수 있는 부분이 있는데, 만약 변수의 개수가 늘어난다면 일일이 다 선언해줘야 할까?
# 이를 해결하기 위해 행렬 곱셈 연산(또는 벡터의 내적)을 사용합니다. => 행렬의 곱셈 과적에서 이루어지는 벡터 연산을 벡터의 내적(Dot Product)이라고 합니다.

# 1. 벡터 연산으로 이해하기
# H(x) = w1x1 + w2x2 + w3x3
# 위의 두 벡터를 각각 X와 W로 표현한다면, 가설은 => H(X) = XW
# x의 개수가 3개였음에도 이제는 X와 W라는 두 개의 변수로 표현된 것을 볼 수 있음.

# 2. 행렬 연산으로 이해하기
# 전체 훈련 데이터의 개수를 셀 수 있는 1개의 단위를 샘플(sample)이라고 합니다. 현재 샘플의 수는 총 5개입니다.
# 각 샘플에서 y를 결정하게 하는 각각의 독립 변수 x를 특성(feature)이라고 합니다. 현재 특성은 3개입니다.
# 이는 종속 변수 x들의 수가 (샘플의 수 × 특성의 수) = 15개임을 의미합니다. 
# 종속 변수 x들을 (샘플의 수 × 특성의 수)의 크기를 가지는 하나의 행렬로 표현해봅시다. 그리고 이 행렬을 X라고 하겠습니다.
# 그리고 여기에 가중치 w1, w2, w3을 원소로 하는 벡터를 W라 하고 이를 곱해보겠습니다.
# Result => H(X) = XW

# Matrix 연산을 고려하여 Pytorch로 구현하기
x_train1  =  torch.FloatTensor([[73,  80,  75], [93,  88,  93], [89,  91,  80], [96,  98,  100], [73,  66,  70]])    # torch.Size([5, 3])
y_train1  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])  # torch.Size([5, 1])
print(x_train1.size(), y_train1.size())   

# W, b 선언
W = torch.zeros((3,1), requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

# W의 크기가 (3 x 1) 벡터 => 행렬의 곱셈이 성립되려면 곱셈의 좌측에 있는 행렬와 우측에 있는 행렬의 행의 크기가 일치해야 함.
# x_trian => (5,3)이며, W 벡터의 크기는 (3 x 1)이므로 두 행렬과 벡터는 행렬곱이 가능합니다.
# hypothesis2 = (x_train1 @ W) + b2
# Hypothesis를 행렬곱으로 정의하였음. 

# optimizer 선언
optimizer2 = optim.SGD([W,b], lr = 1e-5)

EPOCHS2 = 20
for epochs in range(EPOCHS2 + 1):
    # H(x) 계산
    # Bias는 Broadcasting 되어 각 샘플에 더해짐
    hypothesis2 = (x_train1 @ W) + b2
    
    # cost 계산
    cost2 = torch.mean((hypothesis2 - y_train1) **2)

    # cost로 H(x)개선
    optimizer.zero_grad()
    cost2.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epochs, EPOCHS2, hypothesis.squeeze().detach(), cost.item()))

# Epoch    0/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch    1/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch    2/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch    3/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch    4/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch    5/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch    6/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch    7/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch    8/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch    9/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch   10/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch   11/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch   12/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch   13/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch   14/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch   15/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch   16/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch   17/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch   18/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch   19/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378
# Epoch   20/20 hypothesis: tensor([152.4312, 183.9319, 180.8577, 196.9723, 140.4543]) Cost: 1.079378