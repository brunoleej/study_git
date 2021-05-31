import torch

# Broadcasting
m1 = torch.FloatTensor([[3,3]])
m2 = torch.FloatTensor([[2,2]])
print(m1 + m2)  # tensor([[5., 5.]])

# 2 x 1 Vector + 1 x 2 Vector
m3 = torch.FloatTensor([[1,2]])
m4 = torch.FloatTensor([[3],[4]])
print(m3.shape, m4.shape)   # torch.Size([1, 2]) torch.Size([2, 1])
print(m3 + m4)
'''
tensor([[4., 5.],
        [5., 6.]])
'''

# m3
'''
[1, 2]
==> [[1, 2],
     [1, 2]]
'''

# m4
'''
[3]
[4]
==> [[3, 3],
     [4, 4]]
'''

# Matrix Multiplication vs Multiplication
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1],[2]])

print('Shape of Matrix 1 : {}'.format(m1.shape))  # Shape of Matrix 1 : torch.Size([2, 2])
print('Shape of Matrix 2 : {}'.format(m2.shape))  # Shape of Matrix 2 : torch.Size([2, 1])
print(m1 @ m2) # == print(m1.matmul(m2))
'''
tensor([[ 5.],
        [11.]])
'''

m3 = torch.FloatTensor([[1,2], [3, 4]])
m4 = torch.FloatTensor([[1], [2]])

print('Shape of Matrix 3 : {}'.format(m3.shape))  # Shape of Matrix 3 : torch.Size([2, 2])
print('Shape of Matrix 4 : {}'.format(m4.shape))  # Shape of Matrix 4 : torch.Size([2, 1])
print(m3 * m4) # == print(m1.mul(m2))
'''
tensor([[1., 2.],
        [6., 8.]])
'''
