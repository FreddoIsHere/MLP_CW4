from collections import deque
import random
import numpy as np


class Memory:
    def __init__(self):
        self.actions = []
        self.maps = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear_memory(self):
        del self.actions[:]
        del self.maps[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
