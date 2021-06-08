import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

z = torch.rand(3,5, requires_grad=True)
y = torch.randint(5,(3,)).long()

hypothesis = F.softmax(z, dim = 1)

y_one_hot = torch.zeros_like(hypothesis)       # 모든 원소가 0의 값을 가진 3 x 5 텐서를 생성.
print(y_one_hot.scatter_(1, y.unsqueeze(1), 1)) # y.unsqueeze(1)을 하면 (3,)의 크기를 가졌던 y텐서는 (3 x 1)텐서가 됩니다. 

# Low Level
print(torch.log(F.softmax(z, dim = 1)))
'''
tensor([[-1.6240, -1.8566, -2.0208, -1.3913, -1.3266],
        [-1.2238, -1.5293, -1.7906, -1.8792, -1.7742],
        [-1.7135, -1.4338, -1.6957, -1.4240, -1.8505]], grad_fn=<LogBackward>)
'''

# High Level
print(F.log_softmax(z, dim = 1))
'''
tensor([[-1.3892, -1.8023, -1.5992, -1.7922, -1.5271],
        [-1.8887, -1.4585, -1.6572, -1.4954, -1.6030],
        [-1.5322, -1.5237, -1.7035, -1.8264, -1.5007]],
       grad_fn=<LogSoftmaxBackward>)
'''

# F.log_softmax() + F.nll_loss() = F.cross_entropy()
# Low Level
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim = 1).mean()
print(cost) # tensor(1.4689, grad_fn=<MeanBackward0>)

# 두번째 수식
cost2 = (y_one_hot * -F.log_softmax(z, dim = 1)).sum(dim = 1).mean()
print(cost2)    # tensor(1.4689, grad_fn=<MeanBackward0>)

# High Level
# 세 번째 수식
cost3 = F.nll_loss(F.log_softmax(z, dim = 1), y)
print(cost3)    # tensor(1.4689, grad_fn=<NllLossBackward>)

# 네 번째 수식
cost4 = F.cross_entropy(z,y)
print(cost4)    # tensor(1.4689, grad_fn=<NllLossBackward>)

# F.cross_entropy는 비용 함수에 소프트맥스 함수까지 포함하고 있음을 기억!