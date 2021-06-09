# Hight Level
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1, 2, 1, 1],[2, 1, 3, 2],[3, 1, 3, 4],[4, 1, 5, 5],[1, 7, 5, 5],[1, 2, 5, 6],[1, 6, 6, 6],[1, 7, 7, 7]])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])

print(x_train.size())   # torch.Size([8, 4])
print(y_train.size())   # torch.Size([8])

y_one_hot = torch.zeros(8,3)
y_one_hot.scatter_(1, y_train.unsqueeze(1),1)
print(y_one_hot.size()) # torch.Size([8, 3])

W = torch.zeros((4,3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr = 0.1)

EPOCHS = 1000

for epochs in range(EPOCHS + 1):
    hypothesis = x_train @ W + b
    cost = F.cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epochs % 100 == 0:
        print("EPOCHS: {:4d}/{} Cost: {:.6f}".format(epochs,EPOCHS, cost, cost.item()))

'''
EPOCHS:    0/1000 Cost: 1.098612
EPOCHS:  100/1000 Cost: 0.761050
EPOCHS:  200/1000 Cost: 0.689991
EPOCHS:  300/1000 Cost: 0.643229
EPOCHS:  400/1000 Cost: 0.604117
EPOCHS:  500/1000 Cost: 0.568255
EPOCHS:  600/1000 Cost: 0.533922
EPOCHS:  700/1000 Cost: 0.500291
EPOCHS:  800/1000 Cost: 0.466908
EPOCHS:  900/1000 Cost: 0.433507
EPOCHS: 1000/1000 Cost: 0.399962
'''