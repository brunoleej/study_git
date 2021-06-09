import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Data
x_train = torch.FloatTensor([[1, 2, 1, 1],[2, 1, 3, 2],[3, 1, 3, 4],[4, 1, 5, 5],[1, 7, 5, 5],[1, 2, 5, 6],[1, 6, 6, 6],[1, 7, 7, 7]])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])

# Network
class SoftmaxClssifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,3)
    
    def forward(self, x):
        return self.linear(x)

model = SoftmaxClssifierModel()
optimizer = optim.SGD(model.parameters(), lr = 0.1)
EPOCHS = 1000

for epochs in range(EPOCHS + 1):
    hypothesis = model(x_train)
    cost = F.cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epochs % 100 == 0:
        print("EPOCHS:{:4d}/{}, Cost: {:.4f}".format(epochs, EPOCHS,cost.item()))

'''
EPOCHS:   0/1000, Cost: 1.6168
EPOCHS: 100/1000, Cost: 0.6589
EPOCHS: 200/1000, Cost: 0.5734
EPOCHS: 300/1000, Cost: 0.5182
EPOCHS: 400/1000, Cost: 0.4733
EPOCHS: 500/1000, Cost: 0.4335
EPOCHS: 600/1000, Cost: 0.3966
EPOCHS: 700/1000, Cost: 0.3609
EPOCHS: 800/1000, Cost: 0.3254
EPOCHS: 900/1000, Cost: 0.2892
EPOCHS:1000/1000, Cost: 0.2541
'''