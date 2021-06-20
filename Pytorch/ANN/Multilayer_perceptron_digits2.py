import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
Y = digits.target

print("X Type: {}, Y Type: {}".format(X.dtype,Y.dtype)) # X Type: float64, Y Type: int32

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
cost = []

for epochs in range(100):
    optimizer.zero_grad()
    y_pred = model(X)   # forward calculate
    loss = criterion(y_pred, Y)
    loss.backward()
    optimizer.step()

    if epochs % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epochs, 100, loss.item()))
    cost.append(loss.item())
'''
Epoch    0/100 Cost: 2.532156
Epoch   10/100 Cost: 2.156557
Epoch   20/100 Cost: 1.878195
Epoch   30/100 Cost: 1.547330
Epoch   40/100 Cost: 1.170206
Epoch   50/100 Cost: 0.789140
Epoch   60/100 Cost: 0.509978
Epoch   70/100 Cost: 0.345817
Epoch   80/100 Cost: 0.250527
Epoch   90/100 Cost: 0.194673
'''

plt.plot(cost)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost by Epochs')
plt.show()