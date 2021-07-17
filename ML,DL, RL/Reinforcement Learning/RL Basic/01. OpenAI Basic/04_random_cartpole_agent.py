import gym

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    
    observation = env.reset()
    total_reward = 0.0
    total_steps = 0

    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        total_steps += 1

        if done:
            break
    print('Episode done in {}steps, total reward : {:.2f}'.format(total_steps, total_reward))

# Episode done in 19steps, total reward : 19.00
# Episode done in 17steps, total reward : 17.00
# Episode done in 18steps, total reward : 18.00
# Episode done in 21steps, total reward : 21.00
# Episode done in 44steps, total reward : 44.00
