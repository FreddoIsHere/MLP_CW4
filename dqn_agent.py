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
    def __init__(self, env, map_dim, state_dim, action_dim, path="/home/frederik/MLP_CW4", learning_rate=3e-4,
                 gamma=0.99, buffer_size=1000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory = Memory(max_size=buffer_size)
        self.path = path

        try:
            self.actor = torch.load(self.path + "/model.pth")
            self.critic = torch.load(self.path + "/target.pth")
            print("--------------------------------\n"
                  "Models were loaded successfully! \n"
                  "--------------------------------")
        except:
            print("-----------------------\n"
                  "No models were loaded! \n"
                  "-----------------------")
            self.model = Conv_DQN(map_dim, state_dim, action_dim)
            self.target = Conv_DQN(map_dim, state_dim, action_dim)

        self.model_optimizer = torch.optim.Adam(self.model.parameters())
        self.target_optimizer = torch.optim.Adam(self.target.parameters())

    def save(self):
        torch.save(self.model, self.path + "/model.pth")
        torch.save(self.target, self.path + "/target.pth")

    def get_action(self, state, map, epsilon=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        map = torch.FloatTensor(map).unsqueeze(0)
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
    episode_rewards = []

    for e in range(num_episodes):
        state, map = env.reset()
        episode_reward = 0
        tqdm_s = tqdm(range(max_steps), desc = 'Training', leave = True, unit = " steps")
        for step in tqdm_s:
            action = agent.get_action(state, map)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, map, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.memory) > batch_size:
                agent.train(batch_size)

            tqdm_s.refresh()

            if done:
                break

            state = next_state

        print("Episode reward: " + str(episode_reward))
        episode_rewards.append(episode_reward)

    agent.save()
    return episode_rewards


def predict():
    pass


parser = argparse.ArgumentParser(description='DQN_Agent')
parser.add_argument('--num_episodes', nargs="?", type=int, default=100, help='number of episodes')
parser.add_argument('--max_steps', nargs="?", type=int, default=100, help='number of steps')
parser.add_argument('--batch_size', nargs="?", type=int, default=64, help='size of batches')
parser.add_argument('--file', nargs="?", type=str, default='maps', help='file name')
args = parser.parse_args()

env = Map_Environment(args.file, np.array([0, 0, 0]), np.array([50, 50, 50]))
agent = DQN_Agent(env, (50, 50, 50), 3, 6)
rewards = train(env, agent, args.num_episodes, args.max_steps, args.batch_size)
