import gym
import gym_super_mario_bros

# env = gym_super_mario_bros.make('SuperMarioBros<world><level>-n<version>')
# 총 8개의 world로 나누어져 있으며 각 world당 4개의 레벨이 존재합니다.

env = gym_super_mario_bros.make('SuperMarioBros-v0')

# env.reset()
# env.render()
print(env.observation_space.shape)  # (240, 256, 3)  => [height, weight, channel]
print(env.action_space.n)   # 256
