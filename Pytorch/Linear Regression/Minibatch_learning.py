# 전체 데이터를 하나의 행렬로 선언하여 전체 데이터에 대해서 경사 하강법을 수행해 학습 할 수 있지만
# 데이터가 수십만개 이상이라면 전체 데이터에 대해서 경사 하강법을 수행하는 것은 매우 느리고 많은 계산량이 필요함.
# 그렇기 때문에 전체 데이터를 더 작은 단위로 나누어서 해당 단위로 학습하는 개념이 Mini batch라고 합니다.

# Mini-batch 학습을 수행하면 미니 배치만큼만 가져가서 미니 배치에 대한 비용(cost)를 계산하고 경사하강법을 수행합니다.
# 전체 데이터에 대해서 한번에 경사 하강법을 수행하는 방법을 "배치 경사 하강법"이라고 부름.
# 반면 미니 배치 단위로 경사 하강법을 수행하는 방법을 "미니 배치 경사 하강법"이라고 부릅니다.

# 배치 경사 하강법은 경사 하강법을 할 때, 전체 데이터를 사용하므로 가중치 값이 최적값에 수렴하는 과정이 매우 안정적이지만, 계산량이 너무 많이 듭니다. 
# 미니 배치 경사 하강법은 경사 하강법을 할 때, 전체 데이터의 일부만을 보고 수행하므로 최적값으로 수렴하는 과정에서 값이 조금 헤매기도 하지만 훈련 속도가 빠릅니다.
# 배치 크기는 보통 2의 제곱수를 사용합니다. 
# ex) 2, 4, 8, 16, 32, 64... 그 이유는 CPU와 GPU의 메모리가 2의 배수이므로 배치크기가 2의 제곱수일 경우에 데이터 송수신의 효율을 높일 수 있다고 합니다.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# TensorDataset 과 DataLoader를 import
from torch.utils.data import TensorDataset, DataLoader, dataloader  

# TensorDataset은 기본적으로 Tensor를 입력으로 받음.
x_train  =  torch.FloatTensor([[73,  80,  75], [93,  88,  93], [89,  91,  90], [96,  98,  100], [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])


# TensorDataset의 입력으로 사용하고 dataset으로 저장
dataset = TensorDataset(x_train,y_train)

print(list(dataset))

# 데이터로더는 기본적으로 2개의 인자를 입력받음. => 데이터 셋, 미니배치의 크기 (추가적으로 많이 사용되는 인자는 shuffle이 있음)
# shuffle=True를 선택하면 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꿉니다.
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model
model = nn.Linear(3,1)
optimizer = optim.SGD(model.parameters(), lr = 1e-5)


# Hyperparameter
EPOCHS = 20

for epochs in range(EPOCHS + 1):
    for batch_idx, samples in enumerate(dataloader):
        # hypothesis
        hypothesis = model(x_train)

        # cost
        cost = F.mse_loss(hypothesis,y_train)

        # 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(epochs, EPOCHS, batch_idx+1, len(dataloader),cost.item()))

'''
Epoch    0/20 Batch 1/3 Cost: 33144.714844
Epoch    0/20 Batch 2/3 Cost: 10394.444336
Epoch    0/20 Batch 3/3 Cost: 3263.439209
Epoch    1/20 Batch 1/3 Cost: 1028.245850
Epoch    1/20 Batch 2/3 Cost: 327.628754
Epoch    1/20 Batch 3/3 Cost: 108.019791
Epoch    2/20 Batch 1/3 Cost: 39.181667
Epoch    2/20 Batch 2/3 Cost: 17.601915
Epoch    2/20 Batch 3/3 Cost: 10.835341
Epoch    3/20 Batch 1/3 Cost: 8.711813
Epoch    3/20 Batch 2/3 Cost: 8.043674
Epoch    3/20 Batch 3/3 Cost: 7.831722
Epoch    4/20 Batch 1/3 Cost: 7.762733
Epoch    4/20 Batch 2/3 Cost: 7.738574
Epoch    4/20 Batch 3/3 Cost: 7.728483
Epoch    5/20 Batch 1/3 Cost: 7.722795
Epoch    5/20 Batch 2/3 Cost: 7.718443
Epoch    5/20 Batch 3/3 Cost: 7.714565
Epoch    6/20 Batch 1/3 Cost: 7.710820
Epoch    6/20 Batch 2/3 Cost: 7.707148
Epoch    6/20 Batch 3/3 Cost: 7.703469
Epoch    7/20 Batch 1/3 Cost: 7.699771
Epoch    7/20 Batch 2/3 Cost: 7.696126
Epoch    7/20 Batch 3/3 Cost: 7.692416
Epoch    8/20 Batch 1/3 Cost: 7.688792
Epoch    8/20 Batch 2/3 Cost: 7.685102
Epoch    8/20 Batch 3/3 Cost: 7.681459
Epoch    9/20 Batch 1/3 Cost: 7.677789
Epoch    9/20 Batch 2/3 Cost: 7.674121
Epoch    9/20 Batch 3/3 Cost: 7.670482
Epoch   10/20 Batch 1/3 Cost: 7.666800
Epoch   10/20 Batch 2/3 Cost: 7.663179
Epoch   10/20 Batch 3/3 Cost: 7.659493
Epoch   11/20 Batch 1/3 Cost: 7.655868
Epoch   11/20 Batch 2/3 Cost: 7.652210
Epoch   11/20 Batch 3/3 Cost: 7.648584
Epoch   12/20 Batch 1/3 Cost: 7.644897
Epoch   12/20 Batch 2/3 Cost: 7.641257
Epoch   12/20 Batch 3/3 Cost: 7.637611
Epoch   13/20 Batch 1/3 Cost: 7.633992
Epoch   13/20 Batch 2/3 Cost: 7.630362
Epoch   13/20 Batch 3/3 Cost: 7.626752
Epoch   14/20 Batch 1/3 Cost: 7.623086
Epoch   14/20 Batch 2/3 Cost: 7.619423
Epoch   14/20 Batch 3/3 Cost: 7.615847
Epoch   15/20 Batch 1/3 Cost: 7.612221
Epoch   15/20 Batch 2/3 Cost: 7.608632
Epoch   15/20 Batch 3/3 Cost: 7.604958
Epoch   16/20 Batch 1/3 Cost: 7.601335
Epoch   16/20 Batch 2/3 Cost: 7.597726
Epoch   16/20 Batch 3/3 Cost: 7.594120
Epoch   17/20 Batch 1/3 Cost: 7.590465
Epoch   17/20 Batch 2/3 Cost: 7.586861
Epoch   17/20 Batch 3/3 Cost: 7.583271
Epoch   18/20 Batch 1/3 Cost: 7.579674
Epoch   18/20 Batch 2/3 Cost: 7.576089
Epoch   18/20 Batch 3/3 Cost: 7.572456
Epoch   19/20 Batch 1/3 Cost: 7.568830
Epoch   19/20 Batch 2/3 Cost: 7.565250
Epoch   19/20 Batch 3/3 Cost: 7.561612
Epoch   20/20 Batch 1/3 Cost: 7.558069
Epoch   20/20 Batch 2/3 Cost: 7.554402
Epoch   20/20 Batch 3/3 Cost: 7.550841
'''

# 예측값 확인
new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)
print('훈련 후 입력이 73, 80, 75일 때의 예측값 : {}'.format(pred_y))    # 훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[150.4593]], grad_fn=<AddmmBackward>)