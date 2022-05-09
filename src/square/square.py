import warnings
from multiprocessing.sharedctypes import Value
from typing import Dict, Final, List, Literal
from tf_agents.specs import BoundedArraySpec
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import gym
    from gym import error, spaces, utils
    from gym.utils import seeding
    from gym import spaces

import copy
import os
import random
from enum import Enum

import numpy as np

from square import PATH_RESOURCES_FOLDER


class Movement(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class UnknowMovement(Exception):
    pass


class Board:
    def __init__(self, dimension) -> None:
        """ """
        self.dimension: int = 4
        self.board = self.__init_board()

    def __init_board(self):
        """ """
        board = np.zeros((self.dimension, self.dimension), dtype=int)
        return board

    def check_game_over_1(self):
        arr = self.board.ravel()
        arr = self.board[self.board == 0]
        return len(arr) == 0

    def check_game_over_2(self):
        return np.ma.masked_not_equal(self.board, 0).count() == 0

    def get_score(self) -> int:
        """ """
        return np.amax(self.board)

    def new_cell(self):
        """ """
        while True:
            i = random.randint(0, self.board.shape[0] - 1)
            j = random.randint(0, self.board.shape[1] - 1)
            if self.board[i, j] == 0:
                n = random.randint(0, 3)
                self.board[i, j] = 4 if n == 0 else 2
                break

    def process_line(self, line):
        res = []
        for n in line:
            if n == 0:
                # Un 0, on passe.
                continue
            if len(res) == 0:
                # Premier nombre, on ajoute au résultat.
                res.append(n)
            else:
                prev = res[-1]
                if prev == n:
                    # Si le nombre est identique on combine.
                    res[-1] = 2 * n
                else:
                    # Sinon on ajoute.
                    res.append(n)
        while len(res) < len(line):
            res.append(0)
        return res

    def update_game(self, action: Literal[0, 1, 2, 3]) -> bool:
        """ """
        direction = Movement(action)
        is_possible = True
        match direction:
            case Movement.LEFT:
                lines = [
                    self.process_line(self.board[i, :])
                    for i in range(self.board.shape[0])
                ]
                if not np.array_equal(self.board, np.array(lines)):
                    self.board = np.array(lines)
                else:
                    is_possible = False
            case Movement.UP:
                lines = [
                    self.process_line(self.board[:, i])
                    for i in range(self.board.shape[1])
                ]
                if not np.array_equal(self.board, np.array(lines).T):
                    self.board = np.array(lines).T
                else:
                    is_possible = False
            case Movement.RIGHT:
                lines = [
                    list(reversed(self.process_line(self.board[i, ::-1])))
                    for i in range(self.board.shape[0])
                ]
                if not np.array_equal(self.board, np.array(lines)):
                    self.board = np.array(lines)
                else:
                    is_possible = False
            case Movement.DOWN:
                lines = [
                    list(reversed(self.process_line(self.board[::-1, i])))
                    for i in range(self.board.shape[1])
                ]
                if not np.array_equal(self.board, np.array(lines).T):
                    self.board = np.array(lines).T
                else:
                    is_possible = False
            case _:
                raise UnknowMovement

        return is_possible

    def render(self):
        """ """
        print(self.board)


class SquareEnv(gym.Env):
    def __init__(self):
        self.board: Board = Board(dimension=4)
        self.action_space = spaces.Discrete(4)
        lower_bound = np.zeros((4, 4), dtype=int)
        upper_bound = np.matrix(np.ones((4,4)) * np.inf)
        self.observation_space = spaces.Box(lower_bound, upper_bound, dtype=np.int32)    
    

    def render(self, mode="human"):
        """"""
        self.board.render()
        return None

    def step(self, action: Movement):
        """ """
        self.c_step = self.c_step + 1 
        done: bool = False
        is_possible: bool = self.board.update_game(action)

        if is_possible:
            self.board.new_cell()
            

        if self.board.check_game_over_1():
            done = True

        if self.board.check_game_over_2():
            done = True

        if self.c_step >= 100:
            done = True

        observation = self.board.board
        reward = self.board.get_score()
        info = []
        return observation, reward, done, info

    def reset(self, word: str = None):
        self.board: Board = Board(dimension=4)
        self.board.new_cell()
        self.c_step = 0
        return self.board.board
