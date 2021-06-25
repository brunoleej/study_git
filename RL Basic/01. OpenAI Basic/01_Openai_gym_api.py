# The OpenAI Gym API
# The Python library called Gym was developed and has been maintained by OpenAI (www.openai.com). 
# The main goal of Gym is to provide a rich collection of environments for RL experiments using a unified interface.
# So, it it not surprising that the central class in the library is an environment, which is called "Env." Instances of this class expose several methods and fields that provide the required information about its capabilities.
# At a high level, every environment provides these pieces of information and functionality:
    # A set of action that is allowed to be executed in the environment. Gym supports both discrete and continous actions, as well as their combination
    # The shape and boundaries of the observations that the environment provides the agent with
    # A method called "step" to execute an action, which return the current observation, the reward, and the indication that the episode is over
    # A method called "reset", which returns the environment to its initial state and obtains the first obervation

# The Action Space
# As mentioned, the actions that an agent can execute can be discrete, continuous, or a combination of the two. 
# Discrete actions are a fixed set of things that an agent can do, for example, directions in a grid like left, right, up, or down. 
# Another example is a push button, which could be either pressed or released. Both states are mutually exclusive, because a main characteristic of a discrete action space is that only one action from a finite set of actions is possible.

# A continuous action has a value attached to it, for example, a steering wheel, which can be turned at a specific angle, or an accelerator pedal, which can be pressed with different levels of force.
# A description of a continuous action includes the boundaries of the value that the action could have. 
# In the case of a steering wheel, it could be from −720 degrees to 720 degrees. For an accelerator pedal, it's usually from 0 to 1

# Of course, we are not limited to a single action; the environment could take multiple actions, such as pushing multiple buttons simultaneously or steering the wheel and pressing two pedals (the brake and the accelerator). 
# To support such cases, Gym defines a special container class that allows the nesting of several action spaces into one unified action.
