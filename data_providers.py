import pickle
import numpy as np
import random

from collections import deque


class DataProvider(object):

    def __init__(self, file, shuffle_factor=2, seed=123456):
        self.file = open(file, "rb")
        self.shuffle_factor = shuffle_factor
        self.maps = deque(maxlen=50000)
        np.random.seed = seed

    def get_map(self):
        for _ in range(self.shuffle_factor):
            try:
                self.maps.append(pickle.load(self.file))
            except EOFError:
                self.file.close()
                break
        return random.sample(self.maps, 1)

    def __del__(self):
        self.file.close()
