import unittest
import warnings
from typing import Dict, Final, List

from square.square import Movement

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import gym
    from gym import error, spaces, utils
    from gym.utils import seeding
    from gym import spaces

import numpy as np


class TestSquare(unittest.TestCase):

    # def test_board(self):
    #     env = gym.make("Square-v0")
    #     env.reset()
    #     env.render()
    #     while True:
    #         print('====')
    #         action: List[int] = env.action_space.sample()
    #         print(Movement(action))
    #         print(action)
    #         observation, reward, done, info = env.step(Movement(action))
    #         env.render()
    #         if done:
    #             break

    def test_1(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        cube = np.array(
            [
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[20, 21, 22], [23, 24, 25], [26, 27, 28]],
                [[30, 31, 32], [33, 34, 35], [36, 37, 38]],
            ]
        )
        # t = [1,2,3,4,5,6,7,8,9]
        print(matrix[:, 0:2])


if __name__ == "__main__":
    unittest.main()
