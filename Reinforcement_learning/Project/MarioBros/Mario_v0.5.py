import time
import numpy as np
import tensorflow as tf
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from tf_agents.agents import DqnAgent
from gym.wrappers import wrapper

# Build env (first level, right only)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env,RIGHT_ONLY)
env = wrapper(env)

# Prameter
states = (84,84,4)
actions = env.action_space.n

# Agent
agent = DqnAgent()