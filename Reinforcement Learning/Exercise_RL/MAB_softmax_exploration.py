import gym
import gym_bandits
import numpy as np
import math
import random

env = gym.make('BanditTenArmedGaussian-v0')

# number of round (iterations)
num_rounds = 20000

# Count of number of times an arm was pulled
count = np.zeros(10)

# Sum of rewards of each arm
sum_rewards = np.zeros(10)

# Q value which is the average reward
Q = np.zeros(10)

def softmax(tau):
    total = sum([math.exp(val/tau) for val in Q])
    probs = [math.exp(val/tau) / total for val in Q]
    threshold = random.random()
    cumulative_prob = 0.0
    for i in range(len(probs)):
        cumulative_prob += probs[i]
        if (cumulative_prob > threshold):
           return i
    return np.argmax(probs)

for i in range(num_rounds):
    # Select the arm using softmax
    arm = softmax(0.5)
    # Get the reward
    observation, reward, done, info = env.step(arm)
    # update the count of that arm
    count[arm] += 1
    # Sum the rewards obtain from the arm
    sum_rewards[arm]+= reward
    # calculate Q value which is the average reward of the arm
    Q[arm] = sum_rewards[arm] / count[arm]
print('The optimal arm is {}'.format(np.argmax(Q)))