import torch
import torch.nn as nn
import torch.optim as optim

EPOCHS = 10000

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cpu':
    torch.manual_seed(777)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

model = nn.Sequential(
    nn.Linear(2,10),
    nn.Sigmoid(),
    nn.Linear(10,10),
    nn.Sigmoid(),
    nn.Linear(10, 10),
    nn.Sigmoid(),
    nn.Linear(10,1),
    nn.Sigmoid()
).to(device)

criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=1)

for epochs in range(EPOCHS + 1):
    optimizer.zero_grad()
    hypothesis = model(X)

    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if epochs % 100 == 0:
        print('EPOCHS:{:2d}/{}, Cost: {:.4f}'.format(epochs,EPOCHS, cost.item()))
'''
EPOCHS: 0/10000, Cost: 0.6949
EPOCHS:100/10000, Cost: 0.6932
EPOCHS:200/10000, Cost: 0.6932
EPOCHS:300/10000, Cost: 0.6932
EPOCHS:400/10000, Cost: 0.6931
EPOCHS:500/10000, Cost: 0.6931
EPOCHS:600/10000, Cost: 0.6931
EPOCHS:700/10000, Cost: 0.6931
EPOCHS:800/10000, Cost: 0.6931
EPOCHS:900/10000, Cost: 0.6931
EPOCHS:1000/10000, Cost: 0.6931
EPOCHS:1100/10000, Cost: 0.6931
EPOCHS:1200/10000, Cost: 0.6931
EPOCHS:1300/10000, Cost: 0.6931
EPOCHS:1400/10000, Cost: 0.6931
EPOCHS:1500/10000, Cost: 0.6931
EPOCHS:1600/10000, Cost: 0.6931
EPOCHS:1700/10000, Cost: 0.6931
EPOCHS:1800/10000, Cost: 0.6931
EPOCHS:1900/10000, Cost: 0.6931
EPOCHS:2000/10000, Cost: 0.6931
EPOCHS:2100/10000, Cost: 0.6931
EPOCHS:2200/10000, Cost: 0.6931
EPOCHS:2300/10000, Cost: 0.6931
EPOCHS:2400/10000, Cost: 0.6931
EPOCHS:2500/10000, Cost: 0.6931
EPOCHS:2600/10000, Cost: 0.6931
EPOCHS:2700/10000, Cost: 0.6931
EPOCHS:2800/10000, Cost: 0.6931
EPOCHS:2900/10000, Cost: 0.6931
EPOCHS:3000/10000, Cost: 0.6931
EPOCHS:3100/10000, Cost: 0.6931
EPOCHS:3200/10000, Cost: 0.6931
EPOCHS:3300/10000, Cost: 0.6931
EPOCHS:3400/10000, Cost: 0.6930
EPOCHS:3500/10000, Cost: 0.6930
EPOCHS:3600/10000, Cost: 0.6930
EPOCHS:3700/10000, Cost: 0.6930
EPOCHS:3800/10000, Cost: 0.6930
EPOCHS:3900/10000, Cost: 0.6929
EPOCHS:4000/10000, Cost: 0.6929
EPOCHS:4100/10000, Cost: 0.6929
EPOCHS:4200/10000, Cost: 0.6928
EPOCHS:4300/10000, Cost: 0.6927
EPOCHS:4400/10000, Cost: 0.6926
EPOCHS:4500/10000, Cost: 0.6924
EPOCHS:4600/10000, Cost: 0.6921
EPOCHS:4700/10000, Cost: 0.6917
EPOCHS:4800/10000, Cost: 0.6907
EPOCHS:4900/10000, Cost: 0.6886
EPOCHS:5000/10000, Cost: 0.6821
EPOCHS:5100/10000, Cost: 0.6473
EPOCHS:5200/10000, Cost: 0.4530
EPOCHS:5300/10000, Cost: 0.0421
EPOCHS:5400/10000, Cost: 0.0098
EPOCHS:5500/10000, Cost: 0.0050
EPOCHS:5600/10000, Cost: 0.0033
EPOCHS:5700/10000, Cost: 0.0024
EPOCHS:5800/10000, Cost: 0.0019
EPOCHS:5900/10000, Cost: 0.0015
EPOCHS:6000/10000, Cost: 0.0013
EPOCHS:6100/10000, Cost: 0.0011
EPOCHS:6200/10000, Cost: 0.0010
EPOCHS:6300/10000, Cost: 0.0009
EPOCHS:6400/10000, Cost: 0.0008
EPOCHS:6500/10000, Cost: 0.0007
EPOCHS:6600/10000, Cost: 0.0007
EPOCHS:6700/10000, Cost: 0.0006
EPOCHS:6800/10000, Cost: 0.0006
EPOCHS:6900/10000, Cost: 0.0005
EPOCHS:7000/10000, Cost: 0.0005
EPOCHS:7100/10000, Cost: 0.0005
EPOCHS:7200/10000, Cost: 0.0004
EPOCHS:7300/10000, Cost: 0.0004
EPOCHS:7400/10000, Cost: 0.0004
EPOCHS:7500/10000, Cost: 0.0004
EPOCHS:7600/10000, Cost: 0.0003
EPOCHS:7700/10000, Cost: 0.0003
EPOCHS:7800/10000, Cost: 0.0003
EPOCHS:7900/10000, Cost: 0.0003
EPOCHS:8000/10000, Cost: 0.0003
EPOCHS:8100/10000, Cost: 0.0003
EPOCHS:8200/10000, Cost: 0.0003
EPOCHS:8300/10000, Cost: 0.0003
EPOCHS:8400/10000, Cost: 0.0002
EPOCHS:8500/10000, Cost: 0.0002
EPOCHS:8600/10000, Cost: 0.0002
EPOCHS:8700/10000, Cost: 0.0002
EPOCHS:8800/10000, Cost: 0.0002
EPOCHS:8900/10000, Cost: 0.0002
EPOCHS:9000/10000, Cost: 0.0002
EPOCHS:9100/10000, Cost: 0.0002
EPOCHS:9200/10000, Cost: 0.0002
EPOCHS:9300/10000, Cost: 0.0002
EPOCHS:9400/10000, Cost: 0.0002
EPOCHS:9500/10000, Cost: 0.0002
EPOCHS:9600/10000, Cost: 0.0002
EPOCHS:9700/10000, Cost: 0.0002
EPOCHS:9800/10000, Cost: 0.0002
EPOCHS:9900/10000, Cost: 0.0002
EPOCHS:10000/10000, Cost: 0.0002
'''

with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', Y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())