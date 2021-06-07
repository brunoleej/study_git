# H(x) = sigmoid(Wx + b)
# sigmoid의 비용함수를 미분하면 선형회귀 때와는 달리 심한 비볼록 함수(non-convex) 형태의 그래프가 나옴
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.size())   # torch.Size([6, 2])
print(y_train.size())   # torch.Size([6, 1])

W = torch.zeros((2,1), requires_grad=True)  # size => 2 x 1
b = torch.zeros(1, requires_grad=True)

hypothesis = 1 / (1 + torch.exp(-(x_train @ W) + b))

print(hypothesis)
'''
tensor([[0.5000],
        [0.5000],
        [0.5000],
        [0.5000],
        [0.5000],
        [0.5000]], grad_fn=<MulBackward0>)
'''
print(y_train)
'''
tensor([[0.],
        [0.],
        [0.],
        [1.],
        [1.],
        [1.]])
'''

losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
print(losses)
'''
tensor([[0.6931],
        [0.6931],
        [0.6931],
        [0.6931],
        [0.6931],
        [0.6931]], grad_fn=<NegBackward>)
'''

cost = losses.mean()
print(cost) # tensor(0.6931, grad_fn=<MeanBackward0>)

# torch.nn.functional as F => F.binary_cross_entropy(predict_value, real_value)
loss = F.binary_cross_entropy(hypothesis, y_train)
print(loss) # tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)