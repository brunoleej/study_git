import gym

env = gym.make('CartPole-v1')

env.reset()

print("Action Space : {}, Action Space Shape : {}".format(env.action_space, env.action_space.shape)) # Discrete(2) ()
print("Observation Space : {}, Observation Space Shape : {}".format(env.observation_space, env.observation_space.shape))    # Observation Space : Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32), Observation Space Shape : (4,)

print(env.step(0))  # (array([-0.00239458, -0.22985475, -0.02910987,  0.24513798]), 1.0, False, {})
print(env.step(1))  # (array([-0.00699167, -0.03432938, -0.02420711, -0.0565829 ]), 1.0, False, {})
print(env.step(env.action_space.sample()))  # (array([ 0.04516503, -0.15954399,  0.00730418,  0.29586291]), 1.0, False, {})

obs = env.reset()
obs, reward, done, info = env.step(0)
print("Observation : {}, Reward : {}, Done : {}, Info : {}".format(obs, reward, done, info))    # Observation : [ 0.04134092 -0.15451423 -0.05049499  0.23894109], Reward : 1.0, Done : False, Info : {}

'''
Observation:
    Type: Box(4)
    Num     Observation               Min                     Max
    0       Cart Position             -4.8                    4.8
    1       Cart Velocity             -Inf                    Inf
    2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
    3       Pole Angular Velocity     -Inf                    Inf

Actions:
    Type: Discrete(2)
    Num   Action
    0     Push cart to the left
    1     Push cart to the right
'''