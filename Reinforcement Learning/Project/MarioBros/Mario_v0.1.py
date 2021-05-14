import gym
import gym_super_mario_bros

env = gym_super_mario_bros.make('SuperMarioBros-v1')
env.reset()

for _ in range(100000):
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()