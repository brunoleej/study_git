import numpy as np
import torch

# 덮어쓰기 연산
x = torch.FloatTensor([[1, 2], [3, 4]])

# 곱하기 연산을 한 것과 기존의 값 출력
print(x.mul(2.)) # 곱하기 2를 수행한 결과를 출력
'''
tensor([[2., 4.],
        [6., 8.]])
'''
print(x) # 기존의 값 출력
'''
tensor([[1., 2.],
        [3., 4.]])
'''

# 첫번째 출력은 곱하기 2가 수행된 결과를 보여주고, 두번째 출력은 기존의 값이 그대로 출력된 것을 확인할 수 있습니다. 
# 곱하기 2를 수행했지만 이를 x에다가 다시 저장하지 않았으니, 곱하기 연산을 하더라도 기존의 값 x는 변하지 않는 것이 당연합니다.

# 그런데 연산 뒤에 _를 붙이면 기존의 값을 덮어쓰기를 수행함.
print(x.mul_(2.))  # 곱하기 2를 수행한 결과를 변수 x에 값을 저장하면서 결과를 출력
'''
tensor([[2., 4.],
        [6., 8.]])
'''
print(x) # 기존의 값 출력
'''
tensor([[2., 4.],
        [6., 8.]])
'''