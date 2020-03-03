import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class Conv_DQN(nn.Module):

    def __init__(self, map_dim, value_output_dim):
        super(Conv_DQN, self).__init__()
        self.map_dim = map_dim
        self.value_output_dim = value_output_dim

        self.map_net = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv3d(16, 8, kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.feature_input_dim = self.feature_size()

        self.value_net = nn.Sequential(
            nn.Linear(self.feature_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, self.value_output_dim)
        )

    def forward(self, map):
        self.eval()
        map_output = self.map_net(map)
        map_features = map_output.view(map_output.size(0), -1)
        qvals = self.value_net(map_features)
        return qvals

    def feature_size(self):
        return self.map_net(autograd.Variable(torch.zeros(1, *self.map_dim))).view(1, -1).size(1)
