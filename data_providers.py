import pickle
import numpy as np
import random

from collections import deque


class DataProvider(object):

    def __init__(self, file):
        self.file = open(file, "rb")
        self.maps = deque(maxlen=100)

    def get_map(self):
        try:
            self.maps.append(pickle.load(self.file))
        except EOFError:
            self.file.close()
        return self.maps.popleft()

    def __del__(self):
        self.file.close()
