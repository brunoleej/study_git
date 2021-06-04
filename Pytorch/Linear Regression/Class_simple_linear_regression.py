import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import torch.optim as optim

# Random seed
torch.manual_seed(1)

# Data
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

# Simple Linear Regression
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)    # Simple Linear Regression

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# optimizer
optimizer = optim.SGD(model.parameters(),lr = 1e-5)

# Hyperparameter
EPOCHS = 2000

for epochs in range(EPOCHS + 1):
    # hypothesis
    hypothesis = model(x_train)

    # cost
    cost = F.mse_loss(hypothesis,y_train)

    # cost를 통해 hypothesis개선
    # 초기화
    optimizer.zero_grad()
    # compute gradient
    cost.backward()
    # 개선
    optimizer.step()

    if epochs % 100 == 0:
         print('Epoch {:4d}/{} Cost: {:.6f}'.format(epochs, EPOCHS, cost.item()))

'''
Epoch    0/2000 Cost: 13.103541
Epoch  100/2000 Cost: 12.816108
Epoch  200/2000 Cost: 12.534987
Epoch  300/2000 Cost: 12.260030
Epoch  400/2000 Cost: 11.991107
Epoch  500/2000 Cost: 11.728086
Epoch  600/2000 Cost: 11.470836
Epoch  700/2000 Cost: 11.219230
Epoch  800/2000 Cost: 10.973144
Epoch  900/2000 Cost: 10.732457
Epoch 1000/2000 Cost: 10.497054
Epoch 1100/2000 Cost: 10.266816
Epoch 1200/2000 Cost: 10.041631
Epoch 1300/2000 Cost: 9.821383
Epoch 1400/2000 Cost: 9.605979
Epoch 1500/2000 Cost: 9.395290
Epoch 1600/2000 Cost: 9.189227
Epoch 1700/2000 Cost: 8.987686
Epoch 1800/2000 Cost: 8.790568
Epoch 1900/2000 Cost: 8.597774
Epoch 2000/2000 Cost: 8.409211
'''
