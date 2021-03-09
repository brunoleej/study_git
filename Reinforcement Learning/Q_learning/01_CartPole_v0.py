from gym import envs
# env.registry.all()명령을 통해 환경목록을 확인할 수 있습니다
print(envs.registry.all())

# Library 추가
import gym
# make 함수로 simulation instance 생성
env = gym.make('CartPole-v0')
# reset함수를 통해 환경을 초기화
env.reset()
# 반복문을 통해 매 time-step마다 환경을 rendering합니다.
for _ in range(10000):
    env.render()
    env.step(env.action_space.sample())