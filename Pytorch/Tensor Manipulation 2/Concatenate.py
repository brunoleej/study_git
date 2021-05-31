import numpy as np
import torch

# 두개의 Tensor를 연결하는 방법
x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[5,6],[7,8]])

print(x)
print(y)

# 두 텐서를 torch.cat([])을 통해 연결. => 연결 방법은 한 가지만 있는 것이 아님.
# torch.cat은 어느 차원을 늘릴 것인지를 인자로 줄 수 있음. => 예를 들어 dim = 0은 첫번쨰 차원을 늘리라는 의미.
print(torch.cat([x,y], dim = 0))
'''
tensor([[1., 2.],
        [3., 4.],
        [5., 6.],
        [7., 8.]])
'''

# dim = 0을 인자로 했더니 2개의 (2 x 2) Tensor가 (4 x 2)텐서가 된 것을 볼 수 있습니다. 이번에는 dim = 1을 인자로 주겠습니다.
print(torch.cat([x,y],dim = 1))
'''
tensor([[1., 2., 5., 6.],
        [3., 4., 7., 8.]])
'''

# 딥 러닝에서는 주로 모델의 입력 또는 중간 연산에서 두 개의 텐서를 연결하는 경우가 많습니다. 
# 두 텐서를 연결해서 입력으로 사용하는 것은 두 가지의 정보를 모두 사용한다는 의미를 가지고 있습니다.