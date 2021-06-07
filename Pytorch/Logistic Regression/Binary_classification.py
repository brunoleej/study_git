import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# Model Initialize
W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer
optimizer = optim.SGD([W,b], lr = 1)

# Hyperparameter
EPOCHS = 1000

for epochs in range(EPOCHS + 1):
    # hypothesis
    hypothesis = torch.sigmoid(x_train @ W + b)

    # cost
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epochs % 100 == 0:
         print('Epoch {:4d}/{} Cost: {:.6f}'.format(epochs, EPOCHS, cost.item()))

'''
Epoch    0/1000 Cost: 0.693147
Epoch  100/1000 Cost: 0.134722
Epoch  200/1000 Cost: 0.080643
Epoch  300/1000 Cost: 0.057900
Epoch  400/1000 Cost: 0.045300
Epoch  500/1000 Cost: 0.037261
Epoch  600/1000 Cost: 0.031672
Epoch  700/1000 Cost: 0.027556
Epoch  800/1000 Cost: 0.024394
Epoch  900/1000 Cost: 0.021888
Epoch 1000/1000 Cost: 0.019852
'''

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
'''
tensor([[False],
        [False],
        [False],
        [ True],
        [ True],
        [ True]])
'''

print(W)
'''
tensor([[3.2530],
        [1.5179]], requires_grad=True)
'''
print(b)    # tensor([-14.4819], requires_grad=True)
