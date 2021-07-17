import gym

def main():
    env = gym.make('CartPole-v1')
    
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
    print('Episode done in {}steps, Total reward : {:.2f}'.format(total_steps,total_reward))

if __name__ == '__main__':
    main()