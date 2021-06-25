import gym

environmnt = gym.make('CartPole-v1')

observation = environmnt.reset()
print(observation)                  # [-0.00984397  0.0151097  -0.02960235 -0.04028891]

print(environmnt.action_space)      # Discrete(2)
print(environmnt.observation_space) # Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32)

print(environmnt.step(0))           # (array([-0.02978396, -0.20534415,  0.02391799,  0.2534057 ]), 1.0, False, {})
'''
A new observation, which is a new vector of four numbers
A reward of 1.0
The "done" flag with value "False", which means that the episode is not over yet and we are more or less okey
Extra information about the environment, which is an empty dictionary
'''

print(environmnt.action_space.sample())         # 0
print(environmnt.action_space.sample())         # 1
print(environmnt.observation_space.sample())    # [-3.2361224e+00  2.6138890e+38 -2.4192747e-01  1.8180363e+38]
print(environmnt.observation_space.sample())    # [ 2.2867255e+00  2.8260990e+38  3.6918604e-01 -1.0319927e+38]
'''
This method returned a random sample from the underlying space, which in the case of our "Discrete" action space means a random number of 0 or 1, and for the observation space means a random vector of four numbers.

'''