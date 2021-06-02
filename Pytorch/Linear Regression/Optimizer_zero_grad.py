import torch

# 파이토치는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적시키는 특징이 있습니다. 예를 들어봅시다.
w = torch.tensor(2.0, requires_grad=True)

EPOCHS = 20

for epochs in range(EPOCHS + 1):
    z = 2 * w
    z.backward()
    print('수식을 w로 미분한 값 : {}'.format(w.grad))

# 계속해서 미분값인 2가 누적되는 것을 볼 수 있습니다. 그렇기 때문에 optimizer.zero_grad()를 통해 미분값을 계속 0으로 초기화시켜줘야 합니다.