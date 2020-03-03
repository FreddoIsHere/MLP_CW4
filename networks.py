import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Conv_DQN(nn.Module):

    def __init__(self, map_dim, value_output_dim):
        super(Conv_DQN, self).__init__()
        kernel_size = 3

        self.value_net = nn.Sequential(
            nn.Conv3d(in_channels=map_dim[0], out_channels=1, kernel_size=kernel_size, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(map_dim[0]*1*(map_dim[1]-kernel_size+1)*(map_dim[2]-kernel_size+1)*(map_dim[3]-kernel_size+1), 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, value_output_dim)
        )

    def forward(self, map):
        qvals = self.value_net(map)
        return qvals
