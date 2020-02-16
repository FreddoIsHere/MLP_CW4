import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from Memory import Memory

import numpy as np


class DQN_Agent:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = Memory(max_size=buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
