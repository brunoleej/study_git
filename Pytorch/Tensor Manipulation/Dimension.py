import torch

# 1D with Pytorch
t1 = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])

print(t1)                    # tensor([0., 1., 2., 3., 4., 5., 6.])
print(t1.ndim)               # 1
print(t1.shape)              # torch.Size([7])
print(t1.size())             # torch.Size([7])
print(t1[0], t1[1], t1[-1])    # index => tensor(0.) tensor(1.) tensor(6.)
print(t1[2:5], t1[4:-1])      # slicing => tensor([2., 3., 4.]) tensor([4., 5.])
print(t1[:2], t1[3:])         # slicing => tensor([0., 1.]) tensor([3., 4., 5., 6.])


# 2D with Pytorch
t2 = torch.FloatTensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])

print(t2)
'''
tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.]])
'''
print(t2.ndim)      # 2
print(t2.shape)     # torch.Size([4, 3])
print(t2.size())    # torch.Size([4, 3])
print(t2[:, 1])     # row column => tensor([ 2.,  5.,  8., 11.])
print(t2[:, :-1])   # row column 
'''
tensor([[ 1.,  2.],
        [ 4.,  5.],
        [ 7.,  8.],
        [10., 11.]])
'''