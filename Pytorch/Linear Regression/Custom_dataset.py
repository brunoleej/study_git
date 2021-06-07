# torch.utils.data.Dataset을 상속받아 직접 커스텀 데이터셋(Custom Dataset)을 만드는 경우도 있음
# torch.utils.data.Dataset은 pytorch에서 Dataset을 제공하는 추상 클래스임
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset, dataloader
from torch.utils.data import DataLoader

torch.manual_seed(1)

class CustomDataset(Dataset):
    def __init__(self):
        # dataset의 전처리를 해주는 부분
        self.x_data = [[73, 80, 75],[93, 88, 93],[89, 91, 90],[96, 98, 100],[73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    # data의 총 개수를 return
    def __len__(self):
        # dataset의 길이. 즉, 총 샘플의 수를 적어주는 부분
        return len(self.x_data)

    # index를 입력받아 그에 맵핑되는 입출력 데이터를 pytorch의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        # dataset에서 특정 1개의 샘플을 가져오는 함수
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = torch.nn.Linear(3,1)    # input_dim = 3, output_dim = 1
optimizer = optim.SGD(model.parameters(), lr = 1e-5)


EPOCHS = 20

for epochs in range(EPOCHS + 1):
    for batch_idx, samples in enumerate(dataloader):
        # print(batch_idx)
        # print(samples)
        x_train, y_train = samples
        
        # H(x) 계산
        hypothesis = model(x_train)

        # cost
        cost = F.mse_loss(hypothesis, y_train)

        # cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('EPOCHS:{:3d}/{} Batch: {}/{} Cost: {:.6f}'.format(epochs, EPOCHS, batch_idx+1, len(dataloader), cost.item()))

'''
EPOCHS:  0/20 Batch: 1/3 Cost: 28494.531250
EPOCHS:  0/20 Batch: 2/3 Cost: 10967.986328
EPOCHS:  0/20 Batch: 3/3 Cost: 5100.810547
EPOCHS:  1/20 Batch: 1/3 Cost: 682.938171
EPOCHS:  1/20 Batch: 2/3 Cost: 230.594070
EPOCHS:  1/20 Batch: 3/3 Cost: 91.501534
EPOCHS:  2/20 Batch: 1/3 Cost: 22.564547
EPOCHS:  2/20 Batch: 2/3 Cost: 5.264287
EPOCHS:  2/20 Batch: 3/3 Cost: 0.666619
EPOCHS:  3/20 Batch: 1/3 Cost: 0.254771
EPOCHS:  3/20 Batch: 2/3 Cost: 0.412526
EPOCHS:  3/20 Batch: 3/3 Cost: 1.510413
EPOCHS:  4/20 Batch: 1/3 Cost: 0.508034
EPOCHS:  4/20 Batch: 2/3 Cost: 0.028076
EPOCHS:  4/20 Batch: 3/3 Cost: 0.157345
EPOCHS:  5/20 Batch: 1/3 Cost: 0.537085
EPOCHS:  5/20 Batch: 2/3 Cost: 0.217656
EPOCHS:  5/20 Batch: 3/3 Cost: 0.058551
EPOCHS:  6/20 Batch: 1/3 Cost: 0.043198
EPOCHS:  6/20 Batch: 2/3 Cost: 0.588473
EPOCHS:  6/20 Batch: 3/3 Cost: 0.011543
EPOCHS:  7/20 Batch: 1/3 Cost: 0.121354
EPOCHS:  7/20 Batch: 2/3 Cost: 0.603847
EPOCHS:  7/20 Batch: 3/3 Cost: 0.005891
EPOCHS:  8/20 Batch: 1/3 Cost: 0.439329
EPOCHS:  8/20 Batch: 2/3 Cost: 0.191190
EPOCHS:  8/20 Batch: 3/3 Cost: 0.009326
EPOCHS:  9/20 Batch: 1/3 Cost: 0.475450
EPOCHS:  9/20 Batch: 2/3 Cost: 0.245828
EPOCHS:  9/20 Batch: 3/3 Cost: 0.070225
EPOCHS: 10/20 Batch: 1/3 Cost: 0.083207
EPOCHS: 10/20 Batch: 2/3 Cost: 0.005543
EPOCHS: 10/20 Batch: 3/3 Cost: 1.172741
EPOCHS: 11/20 Batch: 1/3 Cost: 0.510159
EPOCHS: 11/20 Batch: 2/3 Cost: 0.045958
EPOCHS: 11/20 Batch: 3/3 Cost: 0.193334
EPOCHS: 12/20 Batch: 1/3 Cost: 0.517951
EPOCHS: 12/20 Batch: 2/3 Cost: 0.305695
EPOCHS: 12/20 Batch: 3/3 Cost: 0.000302
EPOCHS: 13/20 Batch: 1/3 Cost: 0.522243
EPOCHS: 13/20 Batch: 2/3 Cost: 0.227066
EPOCHS: 13/20 Batch: 3/3 Cost: 0.065910
EPOCHS: 14/20 Batch: 1/3 Cost: 0.103471
EPOCHS: 14/20 Batch: 2/3 Cost: 0.578098
EPOCHS: 14/20 Batch: 3/3 Cost: 0.183205
EPOCHS: 15/20 Batch: 1/3 Cost: 0.570738
EPOCHS: 15/20 Batch: 2/3 Cost: 0.050819
EPOCHS: 15/20 Batch: 3/3 Cost: 0.005355
EPOCHS: 16/20 Batch: 1/3 Cost: 0.016758
EPOCHS: 16/20 Batch: 2/3 Cost: 0.544686
EPOCHS: 16/20 Batch: 3/3 Cost: 0.119511
EPOCHS: 17/20 Batch: 1/3 Cost: 0.528579
EPOCHS: 17/20 Batch: 2/3 Cost: 0.221694
EPOCHS: 17/20 Batch: 3/3 Cost: 0.067613
EPOCHS: 18/20 Batch: 1/3 Cost: 0.540206
EPOCHS: 18/20 Batch: 2/3 Cost: 0.063319
EPOCHS: 18/20 Batch: 3/3 Cost: 0.007752
EPOCHS: 19/20 Batch: 1/3 Cost: 0.478905
EPOCHS: 19/20 Batch: 2/3 Cost: 0.243084
EPOCHS: 19/20 Batch: 3/3 Cost: 0.069274
EPOCHS: 20/20 Batch: 1/3 Cost: 0.592551
EPOCHS: 20/20 Batch: 2/3 Cost: 0.107780
EPOCHS: 20/20 Batch: 3/3 Cost: 0.196000
'''

# 임의의 입력 [73, 80, 75] 선언
new_var = torch.FloatTensor([[73, 80, 75]])

# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var)
print('훈련 후 입력이 73, 80, 75 일때의 예측값: ', pred_y)  # 훈련 후 입력이 73, 80, 75 일때의 예측값:  tensor([[150.8774]], grad_fn=<AddmmBackward>)
