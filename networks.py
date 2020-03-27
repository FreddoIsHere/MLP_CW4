import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.distributions import Categorical


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ActorCritic_Net(nn.Module):

    def __init__(self, map_dim, action_output_dim):
        super(ActorCritic_Net, self).__init__()
        kernel_sizes = (2, 3)
        self.conv_net = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=kernel_sizes[0], stride=1),
            nn.ReLU(),
            nn.Conv3d(4, 8, kernel_size=kernel_sizes[1], stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(map_dim[0]*8*(map_dim[1]-(kernel_sizes[0]-1) -(kernel_sizes[1]-1))
                      *(map_dim[2]-(kernel_sizes[0]-1)-(kernel_sizes[1]-1))
                      *(map_dim[3]-(kernel_sizes[0]-1)-(kernel_sizes[1]-1)), 32),
        )

        self.action_net = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_output_dim),
            nn.Softmax(dim=1)
        )

        self.value_net = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

    def forward(self, map): # use for prediction without training
        map = torch.from_numpy(map).float().unsqueeze(0)
        if (torch.cuda.is_available()):
            map = map.cuda()

        state = self.conv_net(map.unsqueeze(0))
        action_probs = self.action_net(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item()

    def act(self, map, memory):
        map = torch.from_numpy(map).float().unsqueeze(0)
        if (torch.cuda.is_available()):
            map = map.cuda()
        memory.maps.append(map)

        state = self.conv_net(map.unsqueeze(0))
        action_probs = self.action_net(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, maps, actions):
        states = self.conv_net(maps)
        action_probs = self.action_net(states)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(torch.squeeze(actions)).unsqueeze(1)
        dist_entropy = dist.entropy().sum(-1).mean()

        state_values = self.value_net(states)

        return action_logprobs, state_values, dist_entropy

