import numpy as np
import torch

# 0으로 채워진 Tensor와 1로 채워진 Tensor
x = torch.FloatTensor([[0,1,2],[2,1,0]])
print(x)
'''
tensor([[0., 1., 2.],
        [2., 1., 0.]])
'''

# 위 텐서에 ones_like를 하면 동일한 크기(shape)지만 1으로만 값이 채워진 텐서를 생성함.
print(torch.ones_like(x))   # 입력 텐서와 크기를 동일하게 하면서 값을 1로 채우기
'''
tensor([[1., 1., 1.],
        [1., 1., 1.]])
'''

# 위 텐서에 zeros_like를 하면 동일한 크기(shape)지만 0으로만 값이 채워진 텐서를 생성함.
print(torch.zeros_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 0으로 채우기
'''
tensor([[0., 0., 0.],
        [0., 0., 0.]])
'''