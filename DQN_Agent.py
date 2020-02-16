import torch
from Networks import Conv_DQN
from Memory import Memory

import numpy as np


class DQN_Agent:
    def __init__(self, env, map_dim, state_dim, action_dim, learning_rate=3e-4, gamma=0.99, buffer_size=50000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = Memory(max_size=buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Conv_DQN(map_dim, state_dim, action_dim).to(self.device)
        self.target = Conv_DQN(map_dim, state_dim, action_dim).to(self.device)

        self.model_optimizer = torch.optim.Adam(self.model.parameters())
        self.target_optimizer = torch.optim.Adam(self.target.parameters())

    def get_action(self, state, map, epsilon=0.1):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state, map)
        action = np.argmax(qvals.cpu().detach().numpy())

        if np.random.randn() < epsilon:
            return self.env.sample()

        return action

    def loss(self):
        pass

    def train(self):
        pass