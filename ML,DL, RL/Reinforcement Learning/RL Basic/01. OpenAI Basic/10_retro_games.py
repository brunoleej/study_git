import retro

env = retro.make(game='Airstriker-Genesis')

if __name__ == "__main__":
    observation = env.reset()
    total_reward = 0.0
    total_steps = 0

    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        total_steps += 1

        if done:
            break
    print('Episode Finished at {}stpes, Toatal Reward : {:.2f}'.format(total_steps, total_reward))
    