# Numpy로 텐서 만들기 (Vector와 Matrix 만들기)
import numpy as np

# 1D with Numpy
# Numpy로 1차원 벡터 생성
t = np.array([0., 1., 2., 3., 4., 5., 6.])

# List를 생성해서 np.array로 1차원 array로 변환함
print(t)    # [0. 1. 2. 3. 4. 5. 6.]

# 1차원 벡터의 차원과 크기를 출력
print('Rank of t : ', t.ndim)   # 1
print('Shape of t : ', t.shape)     # (7,)

# 1차원은 vector, 2차원은 Matrix, 3차원은 3차원 Tensor
# (7,)은 (1,7)을 의미합니다. 다시말해 (1 x 7)의 크기를 가지는 벡터입니다.

# Numpy 기초
print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1])  # 인덱스를 통한 원소 접근
# t[0] t[1] t[-1] =  0.0 1.0 6.0

print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1]) # [시작번호 : 끝 번호]로 범위 지정을 통해 가져온다.
# t[2:5] t[4:-1] =  [2. 3. 4.] [4. 5.]

print('t[:2] t[3:] = ', t[:2], t[3:])   
# t[:2] t[3:] =  [0. 1.] [3. 4. 5. 6.]

# 2D with Numpy
# Numpy로 2차원 행렬을 생성
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)
'''
[[ 1.  2.  3.]
 [ 4.  5.  6.]
 [ 7.  8.  9.]
 [10. 11. 12.]]
'''

print('Rank of t: ', t.ndim)    # 2
print('Shape of t: ', t.shape)  # (4,3)

# Pytorch Tensor Allocation
import torch

# 1D with Pytorch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)    # tensor([0., 1., 2., 3., 4., 5., 6.])

print(t.dim())  # 1
print(t.shape)  # torch.Size([7])
print(t.size()) # torch.Size([7])

# Slicing
print(t[0], t[1], t[-1])    # tensor(0.) tensor(1.) tensor(6.)
print(t[2:5], t[4:-1])  # tensor([2., 3., 4.]) tensor([4., 5.])
print(t[:2], t[3:]) # tensor([0., 1.]) tensor([3., 4., 5., 6.])

# 2D with Pytorch
t = torch.FloatTensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)
'''
tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.]])
'''

# dimension
print(t.dim())  # 2
print(t.size()) # torch.Size([4, 3])

# 현재 텐서의 차원은 2, (4,3)의 크기를 가짐.

print(t[:, 1])  # tensor([ 2.,  5.,  8., 11.])
print(t[:, 1].size())   # torch.Size([4])

print(t[:, :-1])
'''
tensor([[ 1.,  2.],
        [ 4.,  5.],
        [ 7.,  8.],
        [10., 11.]])
'''

# Broadcasting
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)  # tensor([[5., 5.]])


# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # [3] --> [3,3]
print(m1 + m2)  # tensor([[4., 5.]])

# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)
'''
tensor([[4., 5.],
        [5., 6.]])
'''