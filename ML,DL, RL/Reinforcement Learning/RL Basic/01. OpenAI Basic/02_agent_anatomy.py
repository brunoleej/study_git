import random
from typing import List

class Environment:
    def __init__(self):
        self.steps_left = 10
    
    def get_observation(self) -> List[float]:
        return [0.0, 0.0, 0.0]
    
    def get_actions(self) -> List[int]:
        return [0,1]
    
    def is_done(self) -> bool:
        return self.steps_left == 0
    
    def action(self, action: int) -> float:
        if self.is_done():  # if is.done = True gonna make execption
            raise Exception("Game is over") # make Exception
        self.steps_left -= 1
        return random.random()

class Agent:
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env:Environment):
        obs = env.get_observation()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward

if __name__ == "__main__":
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print("Total reward : {:.4f}".format(agent.total_reward))

# 1st trial => Total reward : 4.8448
# 2nd trial => Total reward : 3.8556
# 3rd trial => Total reward : 6.5952
# 4th trial => Total reward : 3.8132