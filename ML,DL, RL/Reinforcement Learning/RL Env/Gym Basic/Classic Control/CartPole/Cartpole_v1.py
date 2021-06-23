import gym

env = gym.make('CartPole-v1')

print("Action Space : {}, Action Space Shape : {}".format(env.action_space, env.action_space.shape)) # Discrete(2) ()
print("Observation Space : {}, Observation Space Shape : {}".format(env.observation_space, env.observation_space.shape))    # Observation Space : Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32), Observation Space Shape : (4,)

observation = env.reset()
for i in range(1000):
    env.render()
    action = env.action_space.sample()

    observation, reward, done, info = env.step(action)
    print("Observation : {}, Reward : {}, Done : {}, Info : {}".format(observation, reward, done, info))

    if done:
        break
env.close()

'''
Observation : [ 0.01600651  0.21722886  0.0080104  -0.30491774], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.02035109  0.41223574  0.00191204 -0.59506366], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.02859581  0.21708708 -0.00998923 -0.30177906], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.03293755  0.41234997 -0.01602481 -0.59759557], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.04118455  0.21745588 -0.02797672 -0.31000306], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.04553366  0.02274347 -0.03417678 -0.02627282], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.04598853  0.21833845 -0.03470224 -0.32953998], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.0503553   0.41393678 -0.04129304 -0.63296124], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.05863404  0.60960973 -0.05395227 -0.93835717], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.07082623  0.41525506 -0.07271941 -0.66310364], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.07913133  0.22121612 -0.08598148 -0.39417513], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.08355566  0.4174462  -0.09386498 -0.71268032], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.09190458  0.22374058 -0.10811859 -0.45095676], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.09637939  0.42021225 -0.11713773 -0.7756684 ], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.10478364  0.22687952 -0.13265109 -0.52201639], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.10932123  0.42359469 -0.14309142 -0.853382  ], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.11779312  0.6203468  -0.16015906 -1.18741938], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.13020006  0.817141   -0.18390745 -1.52571852], Reward : 1.0, Done : False, Info : {}
Observation : [ 0.14654288  0.62465319 -0.21442182 -1.29562031], Reward : 1.0, Done : True, Info : {}
'''
