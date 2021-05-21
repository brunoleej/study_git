import gym
import gym_bandits
import numpy as np

# Initialize Environment
env = gym.make('BanditTenArmedGaussian-v0')

print(env.action_space.n)

# Epsilon-greedy policy
# number of rounds (iterations)
num_rounds = 20000

# Count of number of times an arm was pulled
count = np.zeros(10)

# Sum of rewards of each arm
sum_rewards = np.zeros(10)

# Q value which is the average reward
Q = np.zeros(10)

def epsilon_greedy(epsilon):
    rand = np.random.random()
    if rand < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q)
    return action


for i in range(num_rounds):
    # Select the arm using epsilon greedy
    arm = epsilon_greedy(0.5)
    # Get the reward
    observation, reward, done, info = env.step(arm)
    # update the count of that arm
    count[arm] += 1
    # Sum the rewards obtained from the arm
    sum_rewards[arm] += reward
    # calculate Q value which is the average rewards of the arm
    Q[arm] = sum_rewards[arm] / count[arm]
print('The optimal arm is {}'.format(np.argmax(Q)))