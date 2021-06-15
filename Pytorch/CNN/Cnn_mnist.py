import torch
import torch.nn as nn

# batch_size x channel x height x width
inputs = torch.Tensor(1, 1, 28, 28) 
print('Tensor Shape : {}'.format(inputs.size()))    # Tensor Shape : torch.Size([1, 1, 28, 28])

# Convolution Layer & Pooling
conv1 = nn.Conv2d(1, 32, 3, padding=1) 
print(conv1)    # Conv1d(1, 32, kernel_size=(3,), stride=(1,), padding=same)

conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
print(conv2)    # Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)

# 정수하나를 인자로 넣으면 kernel_size와 stride가 둘 다 해당값으로 지정됨
pool = nn.MaxPool2d(2)
print(pool) # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

flatten = nn.Flatten()
print(flatten)  # Flatten(start_dim=1, end_dim=-1)

# Model
out = conv1(inputs)
print(out.shape)    # torch.Size([1, 32, 28, 28])

out = pool(out)
print(out.shape)    # torch.Size([1, 32, 14, 14])

out = conv2(out)
print(out.shape)    # torch.Size([1, 64, 14, 14])

out = pool(out) 
print(out.shape)    # torch.Size([1, 64, 7, 7])

out = flatten(out)
print(out.shape)    # torch.Size([1, 3136])

fc = nn.Linear(3136, 10)    
out = fc(out)
print(out.shape)    # torch.Size([1, 10])
