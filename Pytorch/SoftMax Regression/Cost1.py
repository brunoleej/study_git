import torch
import torch.nn.functional as F

torch.manual_seed(1)

z = torch.FloatTensor([1,2,3])

hypothesis = F.softmax(z, dim = 0)
print(hypothesis)       # tensor([0.0900, 0.2447, 0.6652])
print(hypothesis.sum()) # tensor(1.)

z1 = torch.rand(3, 5, requires_grad=True)

hypothesis1 = F.softmax(z1, dim = 1)
print(hypothesis1)
'''
tensor([[0.2645, 0.1639, 0.1855, 0.2585, 0.1277],
        [0.2430, 0.1624, 0.2322, 0.1930, 0.1694],
        [0.2226, 0.1986, 0.2326, 0.1594, 0.1868]], grad_fn=<SoftmaxBackward>)
'''

y = torch.randint(5,(3,)).long()
print(y)    # tensor([0, 2, 1])

y_one_hot = torch.zeros_like(hypothesis1)       # 모든 원소가 0의 값을 가진 3 x 5 텐서를 생성.
print(y_one_hot.scatter_(1, y.unsqueeze(1), 1)) # y.unsqueeze(1)을 하면 (3,)의 크기를 가졌던 y텐서는 (3 x 1)텐서가 됩니다. 
'''
tensor([[1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0.]])
'''
print(y.unsqueeze(1))
'''
tensor([[0],
        [2],
        [1]])
'''
# print(y_one_hot.scatter_(1, y.unsqueeze(1), 1)) => y.unsqueeze(1)을 하면 (3,)의 크기를 가졌던 y텐서는 (3 x 1)텐서가 됩니다. 
# scatter의 첫번째 인자로 dim = 1에 대해서 수행하라고 알려주고, 세번째 인자에 숫자 1을 넣어줌으로써 두번째 인자인 y_unsqueeze(1)이 알려주는 위치에 숫자 1을 넣도록 한다.
# 연산 뒤에 _을 붙이면 In-place Operation(덮어쓰기 연산)이다.
print(y_one_hot)
'''
tensor([[1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0.]])
'''

cost = (y_one_hot * -torch.log(hypothesis1)).sum(dim = 1).mean()
print(cost) # tensor(1.4689, grad_fn=<MeanBackward0>)
