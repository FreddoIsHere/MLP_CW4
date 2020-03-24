import numpy as np
import enum
from data_providers import DataProvider
from itertools import product


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

    def __init__(self, max_num_particles, map_file, path_file, shuffle=False):
        self.data_provider = DataProvider(map_file, path_file)
        if shuffle:
            for _ in range(np.random.randint(low=0, high=50)):
                self.data_provider.get_map()
        self.map_dim = (10, 10, 10)
        self.max_num_particles = max_num_particles
        self.lower_bounds = np.array([0, 0, 0])
        self.upper_bounds = np.array([self.map_dim[0]-1, self.map_dim[1]-1, self.map_dim[2]-1])
        self.reset()

    def step(self, action):
        cummulative_reward = 0.0
        N = self.state.shape[0]
        particle_dones = np.zeros(N, dtype=bool)
        for i in range(N):
            particle_done = all(self.state[i] == self.target)
            if ~particle_done:
                self.map[self.state[i, 0], self.state[i, 1], self.state[i, 2]] = 0
                self.state[i], reward = self.execute_action(action, self.state[i])
                reward -= 5*np.sum(np.square(self.state[i] - self.target))
                on_path = (self.state[i] in self.path)
                particle_done = all(self.state[i] == self.target)
                if ~particle_done:
                    self.map[self.state[i, 0], self.state[i, 1], self.state[i, 2]] = -10
                reward += 100*particle_done + 20*on_path
                cummulative_reward += reward
            particle_dones[i] = particle_done
        done = all(particle_dones)
        return self.map, cummulative_reward, done, {}

    def execute_action(self, action, prev_state):
        state = {
            Action.X.value[0]: prev_state + np.array([1, 0, 0]),
            Action.minus_X.value[0]: prev_state + np.array([-1, 0, 0]),
            Action.Y.value[0]: prev_state + np.array([0, 1, 0]),
            Action.minus_Y.value[0]: prev_state + np.array([0, -1, 0]),
            Action.Z.value[0]: prev_state + np.array([0, 0, 1]),
            Action.minus_Z.value[0]: prev_state + np.array([0, 0, -1]),
            Action.XY.value[0]: prev_state + np.array([1, 1, 0]),
            Action.minus_XY.value[0]: prev_state + np.array([-1, -1, 0]),
            Action.XZ.value[0]: prev_state + np.array([1, 0, 1]),
            Action.minus_XZ.value[0]: prev_state + np.array([-1, 0, -1]),
            Action.YZ.value[0]: prev_state + np.array([0, 1, 1]),
            Action.minus_YZ.value[0]: prev_state + np.array([0, -1, -1])
        }[action]
        lower_bound = state < self.lower_bounds
        upper_bound = state > self.upper_bounds
        if any(lower_bound) or any(upper_bound):
            self.obstacle_hit_count += 1
            return prev_state, -5
        elif self.map[state[0], state[1], state[2]] == 1:
            self.obstacle_hit_count += 1
            return prev_state, -5
        return state, -1

    def sample(self):
        idx = np.random.randint(low=0, high=len(Action))
        return idx

    def reset(self):
        self.map, path = self.data_provider.get_map()
        self.start_pos = path[0]
        self.state = np.repeat([self.start_pos], self.max_num_particles, axis=0)
        self.target = path[-1]
        self.path = path[1:]
        combis = product(range(2), repeat=3)
        i = 1
        self.map[self.state[i, 0], self.state[i, 1], self.state[i, 2]] = -10
        for p in combis:
            if i == self.max_num_particles:
                break
            new_state_pos = self.start_pos + np.array(p, dtype=int)
            if np.random.rand() < 0.6 and all(new_state_pos >= [0, 0, 0]) and all(new_state_pos < list(self.map_dim)):
                self.state[i] = new_state_pos
                self.map[self.state[i, 0], self.state[i, 1], self.state[i, 2]] = -10
                i += 1
        self.map[self.target[0], self.target[1], self.target[2]] = -100
        self.obstacle_hit_count = 0
        return self.map


