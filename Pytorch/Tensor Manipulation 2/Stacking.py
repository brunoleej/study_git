import numpy as np
import torch

# 연결(concatenate)을 하는 또 다른 방법으로 스택킹(Stacking)이 있습니다. 스택킹은 영어로 쌓는다는 의미.
# 때로는 연결을 하는 것보다 스택킹이 더 편리할 때가 있는데, 이는 스택킹이 많은 연산을 포함하고 있기 때문.
x = torch.FloatTensor([1,4])
y = torch.FloatTensor([2,5])
z = torch.FloatTensor([3,6])

# torch.stack을 통해 3개의 벡터를 모두 Stacking
print(torch.stack([x,y,z]))
'''
tensor([[1., 4.],
        [2., 5.],
        [3., 6.]])
'''

# 스택킹은 사실 많은 연산을 한 번에 축약하고 있습니다. 예를 들어 위 작업은 아래의 코드와 동일한 작업입니다.
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))

# Stacking에 추가적으로 인자를 줄 수도 있음. => dim = 1을 줌.
# 이는 두번째 차원이 증가하도록 쌓으라는 의미로 해석할 수 있음.
print(torch.stack([x,y,z], dim = 1))
'''
tensor([[1., 2., 3.],
        [4., 5., 6.]])
'''
