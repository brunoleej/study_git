import torch

t1 = torch.FloatTensor([[1, 2], [3,4]])

print(t1)
'''
tensor([[1., 2.],
        [3., 4.]])
'''
print(t1.max()) # tensor(4.)
print(t1.max(dim = 0))
'''
torch.return_types.max(
values=tensor([3., 4.]),
indices=tensor([1, 1]))
'''
print(t1.max(dim = 1))
'''
torch.return_types.max(
values=tensor([2., 4.]),
indices=tensor([1, 1]))
'''

print('Max : ', t1.max(dim = 0)[0])     # Max :  tensor([3., 4.])
print('Argmax : ', t1.max(dim = 0)[1])  # Argmax :  tensor([1, 1])
print(t1.max(dim = 1))
'''
torch.return_types.max(
values=tensor([2., 4.]),
indices=tensor([1, 1]))
'''
print(t1.max(dim = -1))
'''
torch.return_types.max(
values=tensor([2., 4.]),
indices=tensor([1, 1]))
'''