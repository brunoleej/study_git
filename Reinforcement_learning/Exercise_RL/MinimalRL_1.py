# Reinforce_Normal
# Module import
# Reinforce Algorithm은 Episode가 끝나야 계산이 가능함. 왜? => Return이 계산 되어야 하기 때문에
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Policy Class
class Policy(nn.Module):    # nn.Module : 사용자 정의 모듈
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []      # 리스트로 생성
        self.gamma = 0.99   # Discount Factor

        self.fc1 = nn.Linear(4, 128)    # Fully Connected Layer : input이 4차원이니 4차원을 받아서 128차원으로 가는 네트워크 생성
        self.fc2 = nn.Linear(128, 2)    # 128차원을 받아서 최종 2차원으로 가는 네트워크 생성
        self.optimizer = optim.Adam(self.parameters(), lr =0.00005) # Learning_rate 생성

    def forward(self, x):   # Network 생성
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 0)
        return x

    def put_data(self, item):
        self.data.append(item)  
    
    def train(self):
        R = 0
        for r, log_prob in self.data[::-1]:
            R = r + R * self.gamma  # R에서 감가율을 곱하고 그틱의 Reward를 곱함
            loss = -log_prob * R    # Gradient Assent이기 때문에 -를 해줌
            self.optimizer.zero_grad()
            loss.backward() # gradinet가 계산됨(Backpropogation) AutoDiff함수임
            self.optimizer.step()   # 계산된 Gradient 함수를 이용해서 Weight들을 업데이트 해줌
        self.data = []  # 다음 episode에 쌓을 수 있게 비워줌

def main():
    env = gym.make('CartPole-v1')   # env에 CartPole 객체 생성
    pi = Policy()
    avg_t = 0

    for n_epi in range(10000):
        obs = env.reset()   # env의 첫 상태인 Observation 생성
        for t in range(600):
            obs = torch.tensor(obs, dtype = torch.float)   # State가 4차원 벡터임. 원래 ndarray인데 tensor 형태로 변환시켜줌(*tensor로 변환시켜야 Network의 input으로 들어갈 수 있음)
            out = pi(obs)   # pi는 Policy이며 input으로 obs값을 넣습니다. 그게 이제 out값이 됨
            m = Categorical(out)    # Categorical은 pytorch에서 지원하는 확률분포 모델
            action = m.sample()     # tensor(0)이런 방식으로 Sample이 뽑힘
            obs, r, done, info = env.step(action.item())    # action을 던져줌(State Transition하는 줄) 환경에다 Action을 하면 환경이 다음 State와 다음 Reward를 주는 것입니다. item을 하면 텐서의 값을 뽑아줌
            pi.put_data((r,torch.log(out[action])))  # Policy안에 리스트가 있어서 그 Policy안에 데이터를 모아 놓는 것임 어떤걸 집어넣냐? => r(현재 tick의 Reward), 그 tick에 내가 했던 Action의 확률분포 log pi값임
            if done:    # 모든 게임이 끝나거나 엎어지는 순간 True로 들어가는데 게임이 끝난다면
                break   # for문을 멈춥니다.
        avg_t += t
        pi.train()  # pi를 학습시켜라
        if n_epi % 20 == 0 and n_epi != 0:  
            print('# of episode : {}, Avg timestep : {}'.format(n_epi, avg_t / 20.0))    # 현재 학습된 episode의 갯수와 평균적인 타임스텝은 몇 틱을 버텼다라는 것을 출력
            avg_t = 0
    env.close()

if __name__ == '__main__':
    main()

# 다음과 같은 에러가 발생함(그래서 MinimalRL_2로 수정하였음)
# RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [128, 2]], which is output 0 of TBackward, is at version 2; expected version 1 instead. 
# Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).