import numpy as np
import enum
from data_providers import DataProvider


class Action(enum.Enum):
    X = 0,
    minus_X = 1,
    Y = 2,
    minus_Y = 3,
    Z = 4,
    minus_Z = 5,
    XY = 6,
    minus_XY = 7,
    XZ = 8,
    minus_XZ = 9,
    YZ = 10,
    minus_YZ = 11,


class Map_Environment:

    def __init__(self, file, start_pos, target_pos):
        self.data_provider = DataProvider(file)
        self.start_pos = start_pos
        self.state = start_pos
        self.map = self.data_provider.get_map()
        self.map[self.state[0], self.state[1], self.state[2]] = -10
        self.map_dim = self.map.shape
        self.target = target_pos
        self.map[self.target[0], self.target[1], self.target[2]] = -100

    def step(self, action):
        self.map[self.state[0], self.state[1], self.state[2]] = 0
        self.state, reward = self.execute_action(action)
        reward *= np.sum(np.square(self.state - self.target))
        done = all(self.state == self.target)
        reward += 100*done
        self.map[self.state[0], self.state[1], self.state[2]] = -10
        return self.map, reward, done, {}

    def execute_action(self, action):
        state = {
            Action.X.value[0]: self.state + np.array([1, 0, 0]),
            Action.minus_X.value[0]: self.state + np.array([-1, 0, 0]),
            Action.Y.value[0]: self.state + np.array([0, 1, 0]),
            Action.minus_Y.value[0]: self.state + np.array([0, -1, 0]),
            Action.Z.value[0]: self.state + np.array([0, 0, 1]),
            Action.minus_Z.value[0]: self.state + np.array([0, 0, -1]),
            Action.XY.value[0]: self.state + np.array([1, 1, 0]),
            Action.minus_XY.value[0]: self.state + np.array([-1, -1, 0]),
            Action.XZ.value[0]: self.state + np.array([1, 0, 1]),
            Action.minus_XZ.value[0]: self.state + np.array([-1, 0, -1]),
            Action.YZ.value[0]: self.state + np.array([0, 1, 1]),
            Action.minus_YZ.value[0]: self.state + np.array([0, -1, -1])
        }[action]
        lower_bound = state < 0
        upper_bound = state > self.map_dim[0]-1
        obstacle_hit = self.map[self.state[0], self.state[1], self.state[2]] == 1
        if any(lower_bound) or any(upper_bound) or obstacle_hit:
            return self.state, -5
        return state, -1

    def sample(self):
        idx = np.random.randint(low=0, high=len(Action))
        return idx

    def reset(self):
        self.state = self.start_pos
        self.map = self.data_provider.get_map()
        self.map[self.state[0], self.state[1], self.state[2]] = -10
        self.map[self.target[0], self.target[1], self.target[2]] = -100
        return self.map
