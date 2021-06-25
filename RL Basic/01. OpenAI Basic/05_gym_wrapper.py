# Wrappers
# Very frequently, you will want to extend the environment's functionality in some generic way. 
# For example, imagine an environment gives you some observations, but you want to accumulate them in some buffer and provide to the agent the Nlast observations.
# This is a common scenario for dynamic computer games, when one single frame is just not enough to get the full information about the game state. 
# Another example is when you want to be able to crop or preprocess an image's pixels to make it more convenient for the agent to digest, or if you want to normalize reward scores somehow.
# There are many such situations that have the same structure – you want to "wrap" the existing environment and add some extra logic for doing something.
# Gym provides a convenient framework for these situations – the Wrapper class.

# The Wrapper class inherits the Env class. Its constructor accepts the only argument – the instance of the Env class to be "wrapped."
# To add extra functionality, you need to redefine the methods you want to extend, such as step() or reset(). The only requirement is to call the original method of the superclass.

# To handle more specific requirements, such as a Wrapper class that wants to process only observations from the environment, or only actions, there are subclasses of Wrapper that allow the filtering of only a specific portion of information.
# They are as follows:
#     ObservationWrapper: You need to redefine the observation (obs) method of the parent. 
#                         The obs argument is an observation from the wrapped environment, and this method should return the observation that will be given to the agent

#     RewardWrapper: This exposes the reward (rew) method, which can modify the reward value given to the agent.

#     ActionWrapper: You need to override the action (act) method, which can tweak the action passed to the wrapped environment by the agent.

import random
import gym
from typing import TypeVar

Action = TypeVar('Action')

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon = 0):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print('Random')
            return self.env.ation_space.sample()
        return action

if __name__ == "__main__":
    env = RandomActionWrapper(gym.make('CartPole-v1'))
    observation = env.reset()
    total_reward = 0.0

    while True:
        env.render()
        observation, reward, done, info = env.step(0)
        total_reward += reward
        
        if done:
            break
    
    print('Total Reward : {:.2f}'.format(total_reward))

# Total Reward : 9.00
# Total Reward : 10.00