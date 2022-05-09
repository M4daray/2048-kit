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

import tqdm

from square import PATH_RESOURCES_FOLDER


def run_one_episode(env):
    env.reset()
    sum_reward = 0
    while True:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        sum_reward += reward
        if done:
            break
    return sum_reward


if __name__ == "__main__":

    env = gym.make("Square-v0")
    history = []
    for _ in tqdm.tqdm(range(10000)):
        sum_reward = run_one_episode(env)
        history.append(sum_reward)
    avg_sum_reward = sum(history) / len(history)
    print(f"baseline cumulative reward: {avg_sum_reward}")
