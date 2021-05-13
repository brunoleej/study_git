# This API is in beta and may change in the near future.
import torch


# 희소 텐서는 한 쌍의 조밀 한 텐서, 즉 ​​값의 텐서와 인덱스의 2D 텐서로 표시됩니다. 
# 희소 텐서는이 두 개의 텐서와 희소 텐서의 크기 (이 텐서에서 추론 할 수 없음!)를 제공하여 구성 할 수 있습니다. 
# 위치 (0, 2)에 항목 3을 사용하여 희소 텐서를 정의한다고 가정합니다. 
# 위치 (1, 0)에 항목 4, 위치 (1, 2)에 항목 5. 그런 다음 다음과 같이 작성합니다.
i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
v = torch.FloatTensor([3, 4, 5])

print(torch.sparse.FloatTensor(i, v, torch.Size([2,3])).to_dense())
'''
tensor([[0., 0., 3.],
        [4., 0., 5.]])
'''


# LongTensor에 대한 입력은 인덱스 튜플 목록이 아닙니다. 
# 이런 식으로 인덱스를 작성하려면 희소 생성자에 전달하기 전에 전치해야합니다.
i = torch.LongTensor([[0, 2], [1, 0], [1, 2]])
v = torch.FloatTensor([3, 4, 5])

print(torch.sparse.FloatTensor(i.t(), v, torch.Size([2,3])).to_dense())
'''
tensor([[0., 0., 3.],
        [4., 0., 5.]])
'''

# 또한 처음 n 개의 차원 만 희소하고 나머지 차원은 조밀 한 하이브리드 희소 텐서를 생성 할 수도 있습니다.
i = torch.LongTensor([[2, 4]])
v = torch.FloatTensor([[1, 3], [5, 7]])
print(torch.sparse.FloatTensor(i, v).to_dense())
'''
tensor([[0., 0.],
        [0., 0.],
        [1., 3.],
        [0., 0.],
        [5., 7.]])
'''

# 빈 희소 텐서는 크기를 지정하여 생성 할 수 있습니다.
print(torch.sparse.FloatTensor(2, 3))
'''
tensor(indices=tensor([], size=(2, 0)),
       values=tensor([], size=(0,)),
       size=(2, 3), nnz=0, layout=torch.sparse_coo)
'''

# SparseTensor에는 다음과 같은 불변성이 있습니다.

# 1. sparse_dim + density_dim = len (SparseTensor.shape)
# 2. SparseTensor._indices (). shape = (sparse_dim, nnz)
# 3. SparseTensor._values ​​(). shape = (nnz, SparseTensor.shape [sparse_dim :])

# SparseTensor._indices ()는 항상 2D 텐서이므로 가장 작은 sparse_dim = 1입니다.
# 따라서 sparse_dim = 0의 SparseTensor 표현은 단순히 조밀 한 텐서입니다.