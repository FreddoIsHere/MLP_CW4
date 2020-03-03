import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv_Net(nn.Module):

    def __init__(self, map_dim, action_output_dim):
        super(Conv_Net, self).__init__()
        self.map_dim = map_dim
        self.action_output_dim = action_output_dim

        self.action_net = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv3d(16, 8, kernel_size=2, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(8 * self.map_dim[0] * self.map_dim[1] * self.map_dim[2], 32),
            nn.ReLU(),
            nn.Linear(32, self.action_output_dim),
            nn.Softmax(dim=-1)
        )

        self.value_net = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv3d(16, 8, kernel_size=2, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(8 * self.map_dim[0] * self.map_dim[1] * self.map_dim[2], 32),
            nn.ReLU(),
            nn.Linear(32, self.action_output_dim),
        )

    def forward(self, map):
        probs, values = self.action_net(map), self.value_net(map)
        return probs, values
