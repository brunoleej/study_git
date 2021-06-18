import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 32 bit 부동 소수점은 torch.FloatTensor
# 64비트 부호가 있는 정수는 torch.LongTensor를 사용
# GPU 연산을 위한 자료형 => torch.cuda.FloatTensor

x_train = torch.FloatTensor([[1, 2, 1, 1],[2, 1, 3, 2],[3, 1, 3, 4],[4, 1, 5, 5],[1, 7, 5, 5],[1, 2, 5, 6],[1, 6, 6, 6],[1, 7, 7, 7]])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])

print(x_train.size())   # torch.Size([8, 4])
print(y_train.size())   # torch.Size([8])

# Low Level
y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
print(y_one_hot.shape)  # torch.Size([8, 3])

W = torch.zeros((4,3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr = 0.1)

EPOCHS = 1000

for epochs in range(EPOCHS + 1):
    hypothesis = F.softmax(x_train @ W + b)
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim = 1).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epochs % 100 == 0:
         print('Epoch {:4d}/{} Cost: {:.6f}'.format(epochs, EPOCHS, cost.item()))

'''
Epoch    0/1000 Cost: 1.098612
Epoch  100/1000 Cost: 0.761050
Epoch  200/1000 Cost: 0.689991
Epoch  300/1000 Cost: 0.643229
Epoch  400/1000 Cost: 0.604117
Epoch  500/1000 Cost: 0.568255
Epoch  600/1000 Cost: 0.533922
Epoch  700/1000 Cost: 0.500291
Epoch  800/1000 Cost: 0.466908
Epoch  900/1000 Cost: 0.433507
Epoch 1000/1000 Cost: 0.399962
'''