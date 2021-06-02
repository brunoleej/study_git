# Library import
import gym

# Environment 생성
env = gym.make('BipedalWalker-v3')

# Episode 정의
for episode in range(100):
    observation = env.reset()
    for i in range(10000):
        env.render()
        action = env.action_space.sample()
        observation,reward,done,info = env.step(action)
        
        if done:
            print('{} timesteps taken for the Episode'.format(i+1))
            break    