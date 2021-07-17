import gym


if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')

    observation = env.reset()
    total_reward = 0.0
    total_stpes = 0

    while True:
        env.render()
        action = env.action_space.sample()
        observation,reward, done, info = env.step(action)
        total_reward += reward
        total_stpes += 1

        if done:
            break
    
    print('Episode done in : {}stpes, total Reward : {:.2f}'.format(total_stpes, total_reward))

