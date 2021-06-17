import torch
from torch import optim
import torch.nn as nn
from torch.nn.modules.flatten import Flatten
from torch.optim import optimizer
from torch.random import manual_seed
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

lr = 0.001
EPOCHS = 15
batch_size = 100

train = dsets.MNIST(root = 'C:\Study\Pytorch\Dataset', train = True, transform=transforms.ToTensor(), download=True)
test = dsets.MNIST(root = 'C:\Study\Pytorch\Dataset', train = True, transform=transforms.ToTensor(), download=True)

# dataloader를 사용해 batch_size 지정
data_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, drop_last = True)

class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
            )

        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # FC Layer 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   
        out = self.fc(out)
        return out

model = CNN().to(device)

criterion = nn.CrossEntropyLoss().to(device)   
optimizer = optim.Adam(model.parameters(), lr=lr)

total_batch = len(data_loader)
print('total batch_size : {}'.format(total_batch))  # total batch_size : 600

for epoch in range(EPOCHS):
    avg_cost = 0

    for X, Y in data_loader: # X : mini_batch, Y : Label
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))     

with torch.no_grad():
    X_test = test.test_data.view(len(test), 1, 28, 28).float().to(device)
    Y_test = test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy : {}'.format(accuracy.item()))