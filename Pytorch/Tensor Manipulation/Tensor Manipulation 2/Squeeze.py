import numpy as np
import torch

# Squeeze => 1인 차원을 제거
ft = torch.FloatTensor([[0],[1],[2]])
print(ft)
print(ft.shape) # torch.Size([3, 1])
# (3 x 1) => 2번째 dimension이 1이므로 squeeze를 사용하면 (3,)의 크기를 가지는 텐서로 변경됨.

print(ft.squeeze()) # tensor([0., 1., 2.])
print(ft.squeeze().shape)   # torch.Size([3])
# 1이었던 2번째 dimension이 제거되면서 (3,)의 크기를 가지는 텐서로 변경되어 1차원 vector가 되었음.


# Unsqueeze => 특정 위치에 1인 차원을 제거
ft = torch.Tensor([0,1,2])
# print(ft.size())
print(ft.shape) # torch.Size([3])
# 현재는 차원이 1개인 1차원 벡터.
# 1인 차원 추가. => 첫번째 차원의 인덱스를 의미하는 숫자 0을 인자로 넣으면 첫번째 차원에 1인 차원이 추가됨.

print(ft.unsqueeze(0))  # index가 0부터 시작하므로 0은 첫번째 차원을 의미한다. => tensor([[0., 1., 2.]])
print(ft.unsqueeze(0).shape)    # torch.Size([1, 3])

# 위의 결과는 unsqueeze와 view가 동일한 결과를 만든 것을 확인 할 수 있음.
# 이번에는 unsqueeze의 인자를 1로 넣어봄.
# index는 0부터 시작하므로 1은 두번쨰 차원에 1을 추가하겠다는 것을 의미함.
print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)    # torch.Size([3, 1])

# 이번에는 unsqueeze의 인자로 -1을 넣어봄.
# -1은 index상으로 마지막 차원을 의미함.
print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)   # torch.Size([3, 1])