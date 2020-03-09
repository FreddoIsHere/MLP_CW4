import pickle
import numpy as np
import random

from collections import deque


class DataProvider(object):

    def __init__(self, map_file, path_file):
        self.map_file = open(map_file, "rb")
        self.path_file = open(path_file, "rb")
        self.maps = deque(maxlen=100)
        self.paths = deque(maxlen=100)

    def get_map(self):
        try:
            self.maps.append(pickle.load(self.map_file))
            self.paths.append(pickle.load(self.path_file))
        except EOFError:
            print("Maps or paths could not be loaded!")
            self.map_file.close()
            self.path_file.close()
        return self.maps.popleft(), self.paths.popleft()

    def __del__(self):
        self.map_file.close()
        self.path_file.close()
