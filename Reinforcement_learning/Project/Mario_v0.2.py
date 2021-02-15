from nes_py import wrappers
from wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for _ in range(5000):
    if done:
        state = env.reset()
    observation, reward, done, info = env.step(evn.action_space.sample())
    env.render()

env.close()