# torch.unsqueeze()
import torch

x = torch.tensor([1, 2, 3, 4])
print(torch.unsqueeze(x, 0))     # tensor([[1, 2, 3, 4]])

print(torch.unsqueeze(x, 1))
'''
tensor([[1],
        [2],
        [3],
        [4]])
'''