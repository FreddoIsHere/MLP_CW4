import numpy as np
import enum
from data_providers import DataProvider


class Action(enum.Enum):
    X = 1,
    minus_X = 2,
    Y = 3,
    minus_Y = 4,
    Z = 5,
    minus_Z = 6,


class Map_Environment:

    def __init__(self, file, start_pos, target_pos):
        self.data_provider = DataProvider(file)
        self.start_pos = start_pos
        self.state = start_pos
        self.map = self.data_provider.get_map()
        self.target = target_pos

    def step(self, action):
        self.state = self.execute_action(action)
        reward = 0.0
        reward -= np.sum(np.square(self.state - self.target))
        done = self.state == self.target
        if done:
            reward += 10.0
        return self.state, reward, done, {}

    def execute_action(self, action):
        return {
            Action.X: self.state + np.array([1, 0, 0]),
            Action.minus_X: self.state + np.array([-1, 0, 0]),
            Action.Y: self.state + np.array([0, 1, 0]),
            Action.minus_Y: self.state + np.array([0, -1, 0]),
            Action.Z: self.state + np.array([0, 0, 1]),
            Action.minus_Z: self.state + np.array([0, 0, -1])
        }[action]

    def sample(self):
        idx = np.random.randint(low=1, high=len(Action))
        return self.execute_action(Action.values()[idx])

    def reset(self):
        self.state = self.start_pos
        self.map = self.data_provider.get_map()
        return self.state, self.map