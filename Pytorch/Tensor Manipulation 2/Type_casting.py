import numpy as np
import torch
# 텐서에는 자료형이라는 것이 있다. 각 데이터형별로 정의되어져 있는데, 예를 들어 32비트 유동 소수점은 torch.FloatTensor를
# 64비트의 부호 있는 정수는 torch.LongTensor를 사용합니다. GPU 연산을 위한 자료형도 있습니다. 예를 들어 torch.cuda.FloatTensor가 그 예다.
# 그리고 이 자료형을 변환하는 것을 Type Casting이라고 함.

lt = torch.LongTensor([1,2,3,4])
print(lt)   # tensor([1, 2, 3, 4])

# Tensor에다가 .float()을 붙이면 바로 float형으로 타입이 변경됨.
print(lt.float())   # tensor([1., 2., 3., 4.])

# Byte 타입의 bt 텐서 생성
bt = torch.ByteTensor([True, False, False, True])
print(bt)   # tensor([1, 0, 0, 1], dtype=torch.uint8)

# 여기에 .long()이라고하면 long 타입의 텐서로 변경되고 .float()이라고 하면 float 타입의 텐서로 변경된다.
print(bt.long())    # tensor([1, 0, 0, 1])
print(bt.float())   # tensor([1., 0., 0., 1.])
