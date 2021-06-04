import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 3개의 x로부터 하나의 y를 예측하는 문제
# Hypothesis H(x) = w1x1 + w2x2 + w3x3 + b

# Data
x_train = torch.FloatTensor([[73, 80, 75],[93, 88, 93],[89, 91, 90],[96, 98, 100],[73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# Model
model = nn.Linear(3,1)

print(list(model.parameters()))
'''
[Parameter containing:
tensor([[ 0.2975, -0.2548, -0.1119]], requires_grad=True), Parameter containing:
tensor([0.2710], requires_grad=True)]
'''
# W = tensor([[ 0.2975, -0.2548, -0.1119]], requires_grad=True)
# b = tensor([0.2710], requires_grad=True)]

# optimizer
optimizer = optim.SGD(model.parameters(), lr = 1e-5)

# Hyperparameter
EPOCHS = 2000

for epoch in range(EPOCHS + 1):
    # Hypothesis
    hypothesis = model(x_train)
    # model(x_train)은 model.forward(x_train)과 동일함

    # cost
    cost = F.mse_loss(hypothesis, y_train)

    # cost로 Hypothesis 개선
    optimizer.zero_grad()   # gradient를 0으로 초기화
    cost.backward()         # cost를 미분하여 gradient 계산
    optimizer.step()        # W와 b를 업데이트

    if epoch % 100 == 0:
         print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, EPOCHS, cost.item()))

'''
Epoch    0/2000 Cost: 31667.599609
Epoch  100/2000 Cost: 0.225993
Epoch  200/2000 Cost: 0.223911
Epoch  300/2000 Cost: 0.221941
Epoch  400/2000 Cost: 0.220059
Epoch  500/2000 Cost: 0.218271
Epoch  600/2000 Cost: 0.216575
Epoch  700/2000 Cost: 0.214950
Epoch  800/2000 Cost: 0.213413
Epoch  900/2000 Cost: 0.211952
Epoch 1000/2000 Cost: 0.210559
Epoch 1100/2000 Cost: 0.209230
Epoch 1200/2000 Cost: 0.207967
Epoch 1300/2000 Cost: 0.206762
Epoch 1400/2000 Cost: 0.205618
Epoch 1500/2000 Cost: 0.204529
Epoch 1600/2000 Cost: 0.203481
Epoch 1700/2000 Cost: 0.202486
Epoch 1800/2000 Cost: 0.201539
Epoch 1900/2000 Cost: 0.200634
Epoch 2000/2000 Cost: 0.199770
'''

# 임의의 값을 통해 y예측 수행
new_var = torch.FloatTensor([[73,80,75]])

# predict_y에 저장
pred_y = model(new_var)
print('훈련 후 입력이 73, 80, 75일때의 예측값 : ',pred_y)   # 훈련 후 입력이 73, 80, 75일때의 예측값 :  tensor([[151.2306]], grad_fn=<AddmmBackward>)

print(list(model.parameters()))
'''
[Parameter containing:
tensor([[0.9778, 0.4539, 0.5768]], requires_grad=True), Parameter containing:
tensor([0.2802], requires_grad=True)]
'''