import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

model = nn.Sequential(
    nn.Linear(2,1),  # input_dim = 2, output_dim = 1
    nn.Sigmoid()
)

print(model(x_train))
'''
tensor([[0.4020],
        [0.4147],
        [0.6556],
        [0.5948],
        [0.6788],
        [0.8061]], grad_fn=<SigmoidBackward>)
'''

optimizer = optim.SGD(model.parameters(), lr = 1)
EPOCHS = 1000

for epochs in range(EPOCHS + 1):
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epochs % 100 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction)    # 정확도 계산
        print('EPOCHS:{:2d}/{} Cost: {:.6f} Accuracy: {:2.2f}%'.format(epochs, EPOCHS, cost.item(), accuracy * 100))

'''
EPOCHS: 0/1000 Cost: 0.539713 Accuracy: 83.33%
EPOCHS:100/1000 Cost: 0.134272 Accuracy: 100.00%
EPOCHS:200/1000 Cost: 0.080486 Accuracy: 100.00%
EPOCHS:300/1000 Cost: 0.057820 Accuracy: 100.00%
EPOCHS:400/1000 Cost: 0.045251 Accuracy: 100.00%
EPOCHS:500/1000 Cost: 0.037228 Accuracy: 100.00%
EPOCHS:600/1000 Cost: 0.031649 Accuracy: 100.00%
EPOCHS:700/1000 Cost: 0.027538 Accuracy: 100.00%
EPOCHS:800/1000 Cost: 0.024381 Accuracy: 100.00%
EPOCHS:900/1000 Cost: 0.021877 Accuracy: 100.00%
EPOCHS:1000/1000 Cost: 0.019843 Accuracy: 100.00%
'''

print(model(x_train))
'''
tensor([[2.7616e-04],
        [3.1595e-02],
        [3.8959e-02],
        [9.5624e-01],
        [9.9823e-01],
        [9.9969e-01]], grad_fn=<SigmoidBackward>)
'''
print(list(model.parameters()))
'''
[Parameter containing:
tensor([[3.2534, 1.5181]], requires_grad=True), Parameter containing:
tensor([-14.4839], requires_grad=True)]
'''