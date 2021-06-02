import torch

# torch.manual_seed()를 사용한 프로그램의 결과는 다른 컴퓨터에서 실행시켜도 동일한 결과를 얻을 수 있습니다. 
# 그 이유는 torch.manual_seed()는 난수 발생 순서와 값을 동일하게 보장해준다는 특징때문입니다. 
# 우선 랜덤 시드가 3일 때 두 번 난수를 발생시켜보고, 다른 랜덤 시드를 사용한 후에 다시 랜덤 시드를 3을 사용한다면 난수 발생값이 동일하게 나오는지 보겠습니다.

torch.manual_seed(3)
print('Random seed is 3')

for i in range(2):
    print(torch.rand(1))

# First trial (3)
# tensor([0.0043])
# tensor([0.1056])

# Second trial (3)
# tensor([0.0043])
# tensor([0.1056])

# Random seed Change
torch.manual_seed(5)
print('Random seed is 5')

for i in range(2):
    print(torch.rand(1))

# First trial (5)
# tensor([0.8303])
# tensor([0.1261])

# Second trial (5)
# tensor([0.8303])
# tensor([0.1261])

# 텐서에는 requires_grad라는 속성이 있습니다. 이것을 True로 설정하면 자동 미분 기능이 적용됩니다.
# 선형 회귀부터 신경망과 같은 복잡한 구조에서 파라미터들이 모두 이 기능이 적용됩니다. 
# requires_grad = True가 적용된 텐서에 연산을 하면, 계산 그래프가 생성되며 backward 함수를 호출하면 그래프로부터 자동으로 미분이 계산됩니다.