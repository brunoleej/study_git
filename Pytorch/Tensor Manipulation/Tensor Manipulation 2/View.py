import numpy as np
import torch


t = np.array([[[0,1,2], [3,4,5]], [[6,7,8],[9,10,11]]])
ft = torch.FloatTensor(t)

print(ft.shape) # torch.Size([2, 2, 3])

# Transport 3 dimension to 2 dimension
print(ft.view([-1,3]))
print(ft.view([-1,3]).shape)    # torch.Size([4, 3])

# view : 변경전과 변경 후의 텐서 안의 원소의 개수가 유지되어야 합니다.

# Transport to 3 Dimension Tensor
# 3차원텐서에서 차원은 유지하되, Shape을 바꾸는 작업 수행
# (2 x 2 x 3) => (? x 1 x 3) 변경
# (2 x 2 x 3) = (? x 1 x 3) = 12    ? => 4
print(ft.view([-1,1,3]))
print(ft.view([-1,1,3]).shape)  # torch.Size([4, 1, 3])

