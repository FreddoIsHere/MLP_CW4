import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class Conv_DQN(nn.Module):

    def __init__(self, map_dim, state_dim,  value_output_dim):
        super(Conv_DQN, self).__init__()
        self.map_dim = map_dim
        self.state_dim = state_dim
        self.state_ouput_dim = 8
        self.value_output_dim = value_output_dim

        self.map_net = nn.Sequential(
            nn.Conv2d(in_channels=self.map_dim[0], out_channels=64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.state_net = nn.Sequential(
            nn.Linear(self.state_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, self.state_ouput_dim)
        )

        self.feature_input_dim = self.feature_size()

        self.value_net = nn.Sequential(
            nn.Linear(self.feature_input_dim + self.state_ouput_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.value_output_dim)
        )

    def forward(self, state, map):
        map_output = self.map_net(map)
        map_features = map_output.view(map_output.size(0), -1)
        state_ouput = self.state_net(state)
        qvals = self.value_net(torch.cat([state_ouput, map_features], 1))
        return qvals

    def feature_size(self):
        return self.map_net(autograd.Variable(torch.zeros(1, *self.map_dim))).view(1, -1).size(1)
