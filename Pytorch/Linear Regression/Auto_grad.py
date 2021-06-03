import torch
# 경사 하강법 코드를 보고 있으면 requires_grad = True, backward() 등이 나옵니다. 이는 파이토치에서 제공하고 있는 자동 미분(Autograd) 기능을 수행하고 있는 것입니다.
# Gradient Descent Review
# 경사 하강법은 비용 함수를 미분하여 이 함수의 기울기(gradient)룰 구해서 비용이 최소화 되는 방향을 찾아내는 알고리즘.
    # 비용 함수를 손실 함수, 오차 함수라고도 부르므로 비용이 최소화 되는 방향이라는 표현 대신 손실이 최소화 되는 방향 또는 오차를 최소화 되는 방향이라고도 설명할 수 있음.
# 모델이 복잡해질수록 Gradient Descent를 Numpy 등으로 코딩하는 것은 매우 까다롭습니다. 
# Pytorch에서는 Autograd를 사용하면 미분 계산을 자동화하여 Gradient Descent를 손쉽게 사용할 수 있음

w = torch.tensor(2.0, requires_grad=True)

# 수식 정의
y = w**2
z = 2*y + 5

# 위의 수식을 w에 대해서 미분해야 하므로 .backward()를 호출하면 해당 수식의 w에 대한 기울기를 계산합니다.
z.backward()

# w.grad()를 출력하면 w가 속한 수식을 w로 미분한 값이 저장된 것을 확인할 수 있음.
print('수식을 w로 미분한 것 : {}'.format(w.grad))   # 수식을 w로 미분한 것 : 8.0
