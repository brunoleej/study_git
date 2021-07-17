# Monitor
# Another class that you should be aware of is Monitor. 
# It is implemented like Wrapper and can write information about your agent's performance in a file, with an optional video recording of your agent in action.
# Some time ago, it was possible to upload the result of the Monitor class' recording to the https://gym.openai.com website and see your agent's position in comparison to other people's results (see the following screenshot), but, unfortunately, at the end of August 2017, OpenAI decided to shut down this upload functionality and froze all the results.
# There are several alternatives to the original website, but they are not ready yet.
# I hope this situation will be resolved soon, but at the time of writing, it is not possible to check your results against those of others.\

import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, "recording")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))
    env.close()
    env.env.close()