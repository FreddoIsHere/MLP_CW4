import torch
from networks import Conv_DQN
from memory import Memory
import torch.nn.functional as F
from environments import Action
from tqdm import tqdm

import numpy as np


class DQN_Agent:
    def __init__(self, env, map_dim, state_dim, action_dim, path="/home/frederik/MLP_CW4", learning_rate=3e-4,
                 gamma=0.99, tau=1.0, buffer_size=10000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.memory = Memory(max_size=buffer_size)
        self.path = path

        try:
            self.model = torch.load(self.path + "/model.pth")
            self.target = torch.load(self.path + "/target.pth")
            print("--------------------------------\n"
                  "Models were loaded successfully! \n"
                  "--------------------------------")
        except:
            print("-----------------------\n"
                  "No models were loaded! \n"
                  "-----------------------")
            self.model = Conv_DQN(map_dim, state_dim, action_dim)
            self.target = Conv_DQN(map_dim, state_dim, action_dim)

        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.99)

    def save(self):
        torch.save(self.model, self.path + "/model.pth")
        torch.save(self.target, self.path + "/target.pth")

    def get_action(self, map, explore=True, epsilon=0.1):

        if np.random.rand() < epsilon and explore:
            return self.env.sample()

        map = torch.FloatTensor(map).unsqueeze(0).unsqueeze(0)
        qvals = self.model.forward(map)
        action = np.argmax(qvals.detach().numpy())

        return action

    def loss(self, batch):
        maps, actions, rewards, next_maps, dones = batch
        maps = torch.FloatTensor(maps).unsqueeze(1)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_maps = torch.FloatTensor(next_maps).unsqueeze(1)
        dones = torch.BoolTensor(dones)

        # resize tensors
        actions = actions.view(actions.size(0), 1)
        rewards = rewards.view(rewards.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        state_action_values = self.model.forward(maps).gather(1, actions)
        next_state_action_values = torch.max(self.target.forward(next_maps), 1)[0].unsqueeze(1).detach()
        expected_state_action_values = rewards + (~dones) * self.gamma * next_state_action_values
        q_loss = F.mse_loss(state_action_values, expected_state_action_values)

        return q_loss

    def train(self, batch_size):
        batch = self.memory.sample(batch_size)
        model_loss = self.loss(batch)

        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()

        for target_param, param in zip(self.target.parameters(), self.model.parameters()):
            target_param.data = (param.data * self.tau + target_param.data * (1.0 - self.tau)).clone()


def train(env, agent, num_episodes, max_steps, batch_size=64):
    episode_rewards = []

    for e in range(num_episodes):
        map = env.reset()
        episode_reward = 0
        tqdm_s = tqdm(range(max_steps), desc='Training', leave=True, unit=" step")
        for step in tqdm_s:
            action = agent.get_action(map)
            next_map, reward, done, _ = env.step(action)
            agent.memory.push(map, action, reward, next_map, done)
            episode_reward += reward

            if len(agent.memory) > batch_size:
                agent.train(batch_size)

            tqdm_s.refresh()

            if done:
                break

            map = next_map

        print("Episode {} reward: {}".format(e, episode_reward))
        episode_rewards.append(episode_reward)

    agent.save()
    return episode_rewards
