# Library 추가
import gym
# make 함수로 simulation instance 생성
env = gym.make('CarRacing-v0')
# reset함수를 통해 환경을 초기화
env.reset()
# 반복문을 통해 매 time-step마다 환경을 rendering합니다.
for _ in range(10000):
    env.render()
    env.step(env.action_space.sample())