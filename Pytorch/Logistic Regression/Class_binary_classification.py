import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]])
y_train = torch.FloatTensor([[0], [0], [0], [1], [1], [1]])

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()

optimizer = optim.SGD(model.parameters(), lr = 1)

EPOCHS = 1000

for epochs in range(EPOCHS + 1):
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epochs % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format( # 각 에포크마다 정확도를 출력
            epochs, EPOCHS, cost.item(), accuracy * 100,
        ))

'''
Epoch    0/1000 Cost: 0.539713 Accuracy 83.33%
Epoch   10/1000 Cost: 0.614853 Accuracy 66.67%
Epoch   20/1000 Cost: 0.441875 Accuracy 66.67%
Epoch   30/1000 Cost: 0.373145 Accuracy 83.33%
Epoch   40/1000 Cost: 0.316358 Accuracy 83.33%
Epoch   50/1000 Cost: 0.266094 Accuracy 83.33%
Epoch   60/1000 Cost: 0.220498 Accuracy 100.00%
Epoch   70/1000 Cost: 0.182095 Accuracy 100.00%
Epoch   80/1000 Cost: 0.157299 Accuracy 100.00%
Epoch   90/1000 Cost: 0.144091 Accuracy 100.00%
Epoch  100/1000 Cost: 0.134272 Accuracy 100.00%
Epoch  110/1000 Cost: 0.125769 Accuracy 100.00%
Epoch  120/1000 Cost: 0.118297 Accuracy 100.00%
Epoch  130/1000 Cost: 0.111680 Accuracy 100.00%
Epoch  140/1000 Cost: 0.105779 Accuracy 100.00%
Epoch  150/1000 Cost: 0.100483 Accuracy 100.00%
Epoch  160/1000 Cost: 0.095704 Accuracy 100.00%
Epoch  170/1000 Cost: 0.091369 Accuracy 100.00%
Epoch  180/1000 Cost: 0.087420 Accuracy 100.00%
Epoch  190/1000 Cost: 0.083806 Accuracy 100.00%
Epoch  200/1000 Cost: 0.080486 Accuracy 100.00%
Epoch  210/1000 Cost: 0.077425 Accuracy 100.00%
Epoch  220/1000 Cost: 0.074595 Accuracy 100.00%
Epoch  230/1000 Cost: 0.071969 Accuracy 100.00%
Epoch  240/1000 Cost: 0.069526 Accuracy 100.00%
Epoch  250/1000 Cost: 0.067248 Accuracy 100.00%
Epoch  260/1000 Cost: 0.065118 Accuracy 100.00%
Epoch  270/1000 Cost: 0.063122 Accuracy 100.00%
Epoch  280/1000 Cost: 0.061247 Accuracy 100.00%
Epoch  290/1000 Cost: 0.059483 Accuracy 100.00%
Epoch  300/1000 Cost: 0.057820 Accuracy 100.00%
Epoch  310/1000 Cost: 0.056250 Accuracy 100.00%
Epoch  320/1000 Cost: 0.054764 Accuracy 100.00%
Epoch  330/1000 Cost: 0.053357 Accuracy 100.00%
Epoch  340/1000 Cost: 0.052022 Accuracy 100.00%
Epoch  350/1000 Cost: 0.050753 Accuracy 100.00%
Epoch  360/1000 Cost: 0.049546 Accuracy 100.00%
Epoch  370/1000 Cost: 0.048396 Accuracy 100.00%
Epoch  380/1000 Cost: 0.047299 Accuracy 100.00%
Epoch  390/1000 Cost: 0.046252 Accuracy 100.00%
Epoch  400/1000 Cost: 0.045251 Accuracy 100.00%
Epoch  410/1000 Cost: 0.044294 Accuracy 100.00%
Epoch  420/1000 Cost: 0.043376 Accuracy 100.00%
Epoch  430/1000 Cost: 0.042497 Accuracy 100.00%
Epoch  440/1000 Cost: 0.041653 Accuracy 100.00%
Epoch  450/1000 Cost: 0.040843 Accuracy 100.00%
Epoch  460/1000 Cost: 0.040064 Accuracy 100.00%
Epoch  470/1000 Cost: 0.039315 Accuracy 100.00%
Epoch  480/1000 Cost: 0.038593 Accuracy 100.00%
Epoch  490/1000 Cost: 0.037898 Accuracy 100.00%
Epoch  500/1000 Cost: 0.037228 Accuracy 100.00%
Epoch  510/1000 Cost: 0.036582 Accuracy 100.00%
Epoch  520/1000 Cost: 0.035958 Accuracy 100.00%
Epoch  530/1000 Cost: 0.035356 Accuracy 100.00%
Epoch  540/1000 Cost: 0.034773 Accuracy 100.00%
Epoch  550/1000 Cost: 0.034210 Accuracy 100.00%
Epoch  560/1000 Cost: 0.033664 Accuracy 100.00%
Epoch  570/1000 Cost: 0.033137 Accuracy 100.00%
Epoch  580/1000 Cost: 0.032625 Accuracy 100.00%
Epoch  590/1000 Cost: 0.032130 Accuracy 100.00%
Epoch  600/1000 Cost: 0.031649 Accuracy 100.00%
Epoch  610/1000 Cost: 0.031183 Accuracy 100.00%
Epoch  620/1000 Cost: 0.030730 Accuracy 100.00%
Epoch  630/1000 Cost: 0.030291 Accuracy 100.00%
Epoch  640/1000 Cost: 0.029864 Accuracy 100.00%
Epoch  650/1000 Cost: 0.029449 Accuracy 100.00%
Epoch  660/1000 Cost: 0.029046 Accuracy 100.00%
Epoch  670/1000 Cost: 0.028654 Accuracy 100.00%
Epoch  680/1000 Cost: 0.028272 Accuracy 100.00%
Epoch  690/1000 Cost: 0.027900 Accuracy 100.00%
Epoch  700/1000 Cost: 0.027538 Accuracy 100.00%
Epoch  710/1000 Cost: 0.027186 Accuracy 100.00%
Epoch  720/1000 Cost: 0.026842 Accuracy 100.00%
Epoch  730/1000 Cost: 0.026507 Accuracy 100.00%
Epoch  740/1000 Cost: 0.026181 Accuracy 100.00%
Epoch  750/1000 Cost: 0.025862 Accuracy 100.00%
Epoch  760/1000 Cost: 0.025552 Accuracy 100.00%
Epoch  770/1000 Cost: 0.025248 Accuracy 100.00%
Epoch  780/1000 Cost: 0.024952 Accuracy 100.00%
Epoch  790/1000 Cost: 0.024663 Accuracy 100.00%
Epoch  800/1000 Cost: 0.024381 Accuracy 100.00%
Epoch  810/1000 Cost: 0.024104 Accuracy 100.00%
Epoch  820/1000 Cost: 0.023835 Accuracy 100.00%
Epoch  830/1000 Cost: 0.023571 Accuracy 100.00%
Epoch  840/1000 Cost: 0.023313 Accuracy 100.00%
Epoch  850/1000 Cost: 0.023061 Accuracy 100.00%
Epoch  860/1000 Cost: 0.022814 Accuracy 100.00%
Epoch  870/1000 Cost: 0.022572 Accuracy 100.00%
Epoch  880/1000 Cost: 0.022336 Accuracy 100.00%
Epoch  890/1000 Cost: 0.022104 Accuracy 100.00%
Epoch  900/1000 Cost: 0.021877 Accuracy 100.00%
Epoch  910/1000 Cost: 0.021655 Accuracy 100.00%
Epoch  920/1000 Cost: 0.021437 Accuracy 100.00%
Epoch  930/1000 Cost: 0.021224 Accuracy 100.00%
Epoch  940/1000 Cost: 0.021015 Accuracy 100.00%
Epoch  950/1000 Cost: 0.020810 Accuracy 100.00%
Epoch  960/1000 Cost: 0.020609 Accuracy 100.00%
Epoch  970/1000 Cost: 0.020412 Accuracy 100.00%
Epoch  980/1000 Cost: 0.020219 Accuracy 100.00%
Epoch  990/1000 Cost: 0.020029 Accuracy 100.00%
Epoch 1000/1000 Cost: 0.019843 Accuracy 100.00%
'''