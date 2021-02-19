# torch.max( 입력 ) → Tensor
# input텐서 에있는 모든 요소의 최대 값을 반환합니다.
# This function produces deterministic (sub)gradients unlike max(dim=0)
import torch

a = torch.randn(1, 3)
print(a)
# tensor([[ 0.0147, -1.4234, -0.0917]])

print(torch.max(a))
# tensor(0.0147)