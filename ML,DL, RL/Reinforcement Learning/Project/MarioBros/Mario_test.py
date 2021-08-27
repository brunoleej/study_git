import gym_super_mario_bros
import time
import numpy as np
from nes_py.app import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from wrappers import wrapper

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env.RIGHT_ONLY)
env = wrapper(env)

states = (84, 84, 4)
ns = env.action_space.n