import torch

t1 = torch.FloatTensor([[1,2], [3, 4]])

print(t1.sum()) # tensor(10.)
print(t1.sum(dim = 0))  # tensor([4., 6.]) => Sum of 1, 3 and 2, 4
print(t1.sum(dim = 1))  # tensor([3., 7.]) => Sum of 1, 2 and 3, 4
print(t1.sum(dim = -1)) # tensor([3., 7.]) => Sum of 1, 2 and 3, 4
