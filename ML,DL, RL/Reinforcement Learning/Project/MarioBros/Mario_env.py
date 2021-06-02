import gym_super_mario_bros

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env.reset()

print(env.action_space.n)   # 256
print(env.observation_space.shape)  # (240, 256, 3)

for _ in range(10):
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
    # print(observation)
    # print(reward)
    # print(done)
    print(info)
