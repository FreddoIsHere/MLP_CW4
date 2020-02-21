import torch
from networks import Conv_DQN
from memory import Memory
from environments import Map_Environment
import torch.nn.functional as F
import argparse
from environments import Action
from tqdm import tqdm

import numpy as np


class DQN_Agent:
    def __init__(self, env, map_dim, state_dim, action_dim, learning_rate=3e-4, gamma=0.99, buffer_size=50000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory = Memory(max_size=buffer_size)

        self.model = Conv_DQN(map_dim, state_dim, action_dim)
        self.target = Conv_DQN(map_dim, state_dim, action_dim)

        self.model_optimizer = torch.optim.Adam(self.model.parameters())
        self.target_optimizer = torch.optim.Adam(self.target.parameters())

    def get_action(self, state, map, epsilon=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        map = torch.FloatTensor(map)
        qvals = self.model.forward(state, map)
        action = np.argmax(qvals.cpu().detach().numpy())

        if np.random.randn() < epsilon:
            return np.random.randint(low=1, high=len(Action))
        return action

    def loss(self, batch):
        states, maps, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states)
        maps = torch.FloatTensor(maps)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
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
        expected_Q = rewards + (~dones) * self.gamma * next_Q

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
    tqdm_e = tqdm(range(num_episodes), desc='Training', leave=True, unit=" episodes")
    episode_rewards = []

    for e in tqdm_e:
        state, map = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state, map)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, map, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.memory) > batch_size:
                agent.train(batch_size)

            if done:
                break

            state = next_state

        episode_rewards.append(episode_reward)
        tqdm_e.set_description("Episode " + str(e) + ": " + str(episode_reward))
        tqdm_e.refresh()

    return episode_rewards


def predict():
    pass


parser = argparse.ArgumentParser(description='DQN_Agent')
parser.add_argument('--num_episodes', nargs="?", type=int, default=100, help='number of episodes')
parser.add_argument('--max_steps', nargs="?", type=int, default=300, help='number of steps')
parser.add_argument('--batch_size', nargs="?", type=int, default=64, help='size of batches')
parser.add_argument('--file', nargs="?", type=str, default='maps', help='file name')
args = parser.parse_args()

env = Map_Environment(args.file, np.array([0, 0, 0]), np.array([100, 100, 100]))
agent = DQN_Agent(env, (102, 102, 102), 3, 6)
rewards = train(env, agent, args.num_episodes, args.max_steps, args.batch_size)
