import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from torchvision.transforms import transforms

batch_size = 100
EPOCHS = 15

USE_CUDA = torch.cuda.is_available()    # GPU를 사용이 가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu")    # GPU 사용가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다 : {}".format(device))

random.seed(777)
torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# root => Download를 받을 경로
# train => True로 주면 MNIST의 훈련 데이터를 리턴 받으며 False를 주면 테스트 데이터를 리턴 받음
# transform => 현재 데이터를 Pytorch Tensor로 변환해줌
# download => 해당 경로에 MNIST 데이터가 없다면 download를 받겠다는 의미
mnist_train = dset.MNIST(root='C:\Study\Pytorch\Dataset', train = True, transform=transforms.ToTensor(), download = True)    
mnist_test = dset.MNIST(root='C:\Study\Pytorch\Dataset', train = False, transform=transforms.ToTensor(), download = True)

# dataset => load할 대상
# batch_size => batch 크기
# shuffle => 매 Epoch마다 mini batch를 shuffle할 것인지 여부
# drop_last => 마지막 배치를 버릴 것인지 의미
# drop_last를 하는 이유? => 1,000개의 데이터가 있다고 했을 때, 배치 크기가 128이라고 하면. 1,000을 128로 나누면 7개가 나오고 104개가 남는데
# drop_last를 하게 되면 남은 104개를 버리게 됨. => 이는 다른 mini batch보다 개수가 적은 마지막 배치를 경가하강법에 사용하여
# 마지만 배치가 상대적으로 과대 평가되는 현상을 막아줍니다.
data_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last = True)

# Mnist data image of shape 28 * 28 = 784
# to() 함수는 연산을 어디서 수행할 지 결정하는 인자 => CPU는 상관없지만 GPU는 필요함
# bias는 기본 True를 사용
linear = nn.Linear(784, 10, bias=True).to(device)

criterion = nn.CrossEntropyLoss().to(device)    # 내부적으로 softmax 함수를 포함하고 있음
optimizer = optim.SGD(linear.parameters(), lr = 0.1)

for epochs in range(EPOCHS):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        # batch_size가 100이므로 아래의 연산에서 X는 (100,784)의 텐서가 된다.
        X = X.view(-1, 28*28).to(device)
        # Label은 One-hot Encoding이 된 상태가 아니라 0 ~ 9의 정수
        Y = Y.to(device)

        hypothesis = linear(X)
        cost = criterion(hypothesis,Y)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('EPOCHS:{:4d}/{}, Loss: {:.4f}'.format(epochs + 1, EPOCHS, avg_cost))
print('Learning Finished')        

'''
EPOCHS:   1/15, Loss: 0.5349
EPOCHS:   2/15, Loss: 0.3593
EPOCHS:   3/15, Loss: 0.3311
EPOCHS:   4/15, Loss: 0.3166
EPOCHS:   5/15, Loss: 0.3071
EPOCHS:   6/15, Loss: 0.3002
EPOCHS:   7/15, Loss: 0.2949
EPOCHS:   8/15, Loss: 0.2908
EPOCHS:   9/15, Loss: 0.2874
EPOCHS:  10/15, Loss: 0.2846
EPOCHS:  11/15, Loss: 0.2818
EPOCHS:  12/15, Loss: 0.2799
EPOCHS:  13/15, Loss: 0.2778
EPOCHS:  14/15, Loss: 0.2760
EPOCHS:  15/15, Loss: 0.2744
'''