
# 자동미분 패키지인 Autograd
import torch

# requries_grad = True를 설정하여 연산을 기록한다.
x = torch.ones(2,2, requires_grad=True)  # 요소 값이 1인 2x2행렬을 선언한다.
print(x)
'''
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
'''

# Tensor에 연산을 수행한다. requires_grad = True를 설정했기 때문에 연산을 기록한다.
y = x + 2
print(y)
'''
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)  
'''
# y는 연산의 결과로 생성된 것이므로 grad_fn을 갖는다.

# grad_fn이 생성된다.
print(y.grad_fn)
# <AddBackward0 object at 0x7f9fde660b20>

# y에 다른 연산을 수행한다.
z = y * y * 3
out = z.mean()

print(z, out)
'''
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)
'''
#.requires_grad(...)는 기존 Tensor의 requires_grad 값을 바꿔치기 (in-place)하여 변경한다. 입력값이 지정되지 않으면 기본값은 False이다.

# 예제
a = torch.randn(2, 2)  # standard normal distribution (2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
# False

a.requires_grad_(True)  # .requires_grad(...)는 기존 Tensor의 requires_grad 값을 바꿔치기(in-place)하여 변경한다.
print(a.requires_grad)
# True

b = (a * a).sum()
print(b.grad_fn)  # a.requires_grad_(True)로 변경해줬기 떄문에 이제 추적이 되는거다.
# <SumBackward0 object at 0x7fe202768b20>

# requires_grad(...) 함수가 False일 때와 True일 때, 다른점?
# requires_grad=True를 설정하면 연산을 기록할 수 있다. True일 때, 연산을 기록할 수 있다면 False일 때는 연산을 기록할 수 없다는 것

# grad_fn은 무엇?
#requires_grad=True로 설정한 변수로 다른 연산을 할 경우 생성되는 것 같다.(확실하지는 않다.)




