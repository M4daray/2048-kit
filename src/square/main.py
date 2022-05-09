import warnings
from typing import Dict, Final, List

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import gym
    from gym import error, spaces, utils
    from gym.utils import seeding
    from gym import spaces

import copy
import os
import random

from square import PATH_RESOURCES_FOLDER

if __name__ == "__main__":

    env = gym.make("Square-v0")
    env.reset()
    env.render()
    while True:
        action: List[int] = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            break
