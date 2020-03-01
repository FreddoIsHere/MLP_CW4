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


class Map_Environment:

    def __init__(self, file, start_pos, target_pos):
        self.data_provider = DataProvider(file)
        self.start_pos = start_pos
        self.state = start_pos
        self.map = self.data_provider.get_map()
        self.map[self.state[0], self.state[1], self.state[2]] = -1
        self.map_dim = self.map.shape
        self.target = target_pos

    def step(self, action):
        self.map[self.state[0], self.state[1], self.state[2]] = 0
        self.state = self.execute_action(action)
        reward = 0.0
        reward -= np.sum(np.square(self.state - self.target))
        #obstacle_hit = self.map[self.state[0], self.state[1], self.state[2]]
        #reward -= 100*obstacle_hit
        done = all(self.state == self.target)
        reward += 1000*done
        self.map[self.state[0], self.state[1], self.state[2]] = -1
        return self.map, reward, done, {}

    def execute_action(self, action):
        state = {
            Action.X.value[0]: self.state + np.array([1, 0, 0]),
            Action.minus_X.value[0]: self.state + np.array([-1, 0, 0]),
            Action.Y.value[0]: self.state + np.array([0, 1, 0]),
            Action.minus_Y.value[0]: self.state + np.array([0, -1, 0]),
            Action.Z.value[0]: self.state + np.array([0, 0, 1]),
            Action.minus_Z.value[0]: self.state + np.array([0, 0, -1])
        }[action]
        mask = state < 0
        mask2 = state > self.map_dim[0]-1
        state[mask] = 0
        state[mask2] = self.map_dim[0]-1
        return state

    def sample(self):
        idx = np.random.randint(low=0, high=len(Action))
        return self.execute_action(idx)

    def reset(self):
        self.state = self.start_pos
        self.map = self.data_provider.get_map()
        self.map[self.state[0], self.state[1], self.state[2]] = -1
        return self.map
