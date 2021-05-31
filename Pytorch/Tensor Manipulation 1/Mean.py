import torch

t1 = torch.FloatTensor([1,2])
print(t1.mean()) # tensor(1.5000)

t2 = torch.FloatTensor([[1,2], [3,4]])
print(t2.mean())    # tensor(2.5000)
print(t2.mean(dim = 0)) # dim=0 => first dimension(row)
# Mean of 1, 3 and 1, 4

print(t2.mean(dim = 1)) # tensor([1.5000, 3.5000])
# Mean of 1,2 and 3, 4

print(t2.mean(dim = -1))    # tensor([1.5000, 3.5000])
# Mean of 1,2 and 3, 4
