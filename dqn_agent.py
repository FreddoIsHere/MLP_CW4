import torch
from networks import Conv_DQN
from memory import Memory
import torch.nn.functional as F
from environments import Action
from tqdm import tqdm

import numpy as np


class DQN_Agent:
    def __init__(self, env, map_dim, action_dim, path="/home/frederik/MLP_CW4", learning_rate=5e-5,
                 gamma=0.999, tau=0.5, buffer_size=50000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_decay = 0.995
        self.epsilon = 0.3
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
            self.model = Conv_DQN(map_dim, action_dim)
            self.target = Conv_DQN(map_dim, action_dim)

        self.update_counter = 0
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-3, weight_decay=0.999)

    def save(self):
        torch.save(self.model, self.path + "/model.pth")
        torch.save(self.target, self.path + "/target.pth")

    def get_action(self, map, explore=True):

        if np.random.rand() < self.epsilon and explore:
            return self.env.sample()

        map = torch.FloatTensor(map).unsqueeze(0).unsqueeze(0)
        qvals = self.model.forward(map).detach()
        action = np.argmax(qvals.numpy())

        return action

    def loss(self, batch):
        maps, actions, rewards, next_maps, dones = batch
        maps = torch.FloatTensor(maps).unsqueeze(1)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_maps = torch.FloatTensor(next_maps).unsqueeze(1)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        state_action_values = self.model.forward(maps).gather(1, actions)
        next_state_action_values = torch.max(self.target.forward(next_maps), 1)[0].unsqueeze(1).detach()
        expected_state_action_values = rewards + (~dones) * self.gamma * next_state_action_values
        q_loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.model_optimizer.zero_grad()
        q_loss.backward()
        self.model_optimizer.step()

        return q_loss

    def train(self, batch_size):
        self.update_counter += 1
        batch = self.memory.sample(batch_size)
        model_loss = self.loss(batch)

        #for target_param, param in zip(self.target.parameters(), self.model.parameters()):
            #target_param.data = (param.data * self.tau + target_param.data * (1.0 - self.tau)).clone()

        if self.update_counter % 3 == 0:
            self.target.load_state_dict(self.model.state_dict())

        return model_loss


def train(env, agent, num_episodes, max_steps, batch_size=32):
    episode_rewards = []
    episode_losses = []
    tqdm_e = tqdm(range(num_episodes), desc='Training', leave=True, unit="episode")
    for e in tqdm_e:
        map = env.reset()
        episode_reward = 0
        episode_loss = 0
        for step in range(max_steps):
            action = agent.get_action(map)
            next_map, reward, done, _ = env.step(action)
            agent.memory.push(map, action, reward, next_map, done)
            episode_reward += reward

            if len(agent.memory) > batch_size:
                loss = agent.train(batch_size)
                #print(loss)
                episode_loss += loss.detach().numpy()

            if done:
                print("Target reached: ", episode_reward)
                break

            map = next_map

        agent.epsilon *= agent.epsilon_decay
        tqdm_e.set_description("Episode {} reward: {} pos: {} ep: {}".format(e, round(episode_reward), env.state, np.round(agent.epsilon, decimals=2)))
        tqdm_e.refresh()
        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss/max_steps)

    agent.save()
    return episode_rewards, episode_losses
