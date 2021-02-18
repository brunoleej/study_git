import torch

# ==============================================================================
# torch.gather(input, dim, inde, * , sparse_grad = False, out = None) -> Tensor

# 3차원 텐서의 경우 출력은 다음과 같이 지정
# out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
# out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
# out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

# Parameters
# - input (Tensor) – 텐서의 Source
# - dim (int) – 인덱싱 할 축
# - index(LongTensor) - 수집할 요소의 인덱스
# - sparse_grad(bool, optional) - 만약 True이면 Gradient w, r, t input은 sparse tensor가 됩니다.
# - out(Tensor, optional) - the destination tensor

t = torch.tensor([[1,2],[3,4]])
print(torch.gather(t, 1, torch.tensor([[0,0],[1,0]])))

'''
tensor([[1, 1],
        [4, 3]])
'''