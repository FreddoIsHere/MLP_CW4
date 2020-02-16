import torch
from networks import Conv_DQN
from memory import Memory
import torch.nn.functional as F

import numpy as np


class DQN_Agent:
    def __init__(self, env, map_dim, state_dim, action_dim, learning_rate=3e-4, gamma=0.99, buffer_size=50000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory = Memory(max_size=buffer_size)

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

    def loss(self, batch):
        states, maps, actions, rewards, next_states, dones = batch
        states = torch.IntTensor(states).to(self.device)
        maps = torch.IntTensor(maps).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.IntTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones)

        # resize tensors
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        # compute loss
        model_Q = self.model.forward(states, maps).gather(1, actions)
        target_Q = self.target.forward(states, maps).gather(1, actions)

        next_model_Q = self.model.forward(next_states, maps)  # only if map doesn't change within episode
        next_target_Q = self.target.forward(next_states, maps)
        next_Q = torch.min(
            torch.max(next_model_Q, 1)[0],
            torch.max(next_target_Q, 1)[0]
        )
        next_Q = next_Q.view(next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * next_Q

        model_loss = F.mse_loss(model_Q, expected_Q.detach())
        target_loss = F.mse_loss(target_Q, expected_Q.detach())

        return model_loss, target_loss

    def train(self, batch_size):
        batch = self.memory.sample(batch_size)
        model_loss, target_loss = self.loss(batch)

        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()

        self.target_optimizer.zero_grad()
        target_loss.backward()
        self.target_optimizer.step()


def train(env, agent, num_episodes, max_steps, batch_size=64):
    episode_rewards = []

    for episode in range(num_episodes):
        state, map = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state, map)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, map, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.memory) > batch_size:
                agent.train(batch_size)

            if done:
                break

            state = next_state

        episode_rewards.append(episode_reward)
        print("Episode " + str(episode) + ": " + str(episode_reward))

    return episode_rewards


def predict():
    pass
