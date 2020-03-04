import torch
from networks import ActorCritic_Net
from memory import Memory
import torch.nn.functional as F
from environments import Action
from tqdm import tqdm

import numpy as np
from torch.distributions import Categorical
import torch.nn as nn


class PPO_Agent:
    def __init__(self, env, map_dim, action_dim, path="/home/frederik/MLP_CW4", learning_rate=2e-4,
                 gamma=0.99, eps_clip=0.2):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.memory = Memory()
        self.path = path

        try:
            self.policy = torch.load(self.path + "/model.pth")
            print("--------------------------------\n"
                  "Models were loaded successfully! \n"
                  "--------------------------------")
        except:
            print("-----------------------\n"
                  "No models were loaded! \n"
                  "-----------------------")
            self.policy = ActorCritic_Net(map_dim, action_dim)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.old_policy = ActorCritic_Net(map_dim, action_dim)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.loss = nn.MSELoss()

    def save(self):
        torch.save(self.policy, self.path + "/model.pth")

    def update(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            discounted_reward = reward + (~done)*self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)

        # convert list to tensor
        maps = torch.stack(self.memory.maps).detach()
        actions = torch.LongTensor(self.memory.actions).unsqueeze(1).detach()
        logprobs = torch.FloatTensor(self.memory.logprobs).unsqueeze(1).detach()

        # Evaluating old actions and values :
        next_logprobs, next_state_values, next_entropy = self.policy.evaluate(maps, actions)

        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(next_logprobs - logprobs.detach())

        # Finding Surrogate Loss:
        advantages = rewards - next_state_values.detach()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        loss = policy_loss + 0.5 * self.loss(rewards, next_state_values) - 0.001 * next_entropy
        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        # Copy new weights into old policy:
        self.old_policy.load_state_dict(self.policy.state_dict())

        return policy_loss.detach().numpy()


def train(env, agent, num_episodes, max_steps):
    episode_rewards = []
    episode_losses = []
    target_reached = 0
    tqdm_e = tqdm(range(num_episodes), desc='Training', leave=True, unit="episode")
    for e in tqdm_e:
        map = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.old_policy.act(map, agent.memory)
            next_map, reward, done, _ = env.step(action)
            agent.memory.rewards.append(reward)
            agent.memory.dones.append(done)
            episode_reward += reward

            if done:
                target_reached += 1
                break

            map = next_map

        loss = agent.update()
        agent.memory.clear_memory()

        if (e+1) % 51 == 0:
            tqdm_e.set_description("Epi {} avg_r: {} Reached: {}".format(e, np.mean(episode_rewards[-50]), target_reached))
        tqdm_e.refresh()
        episode_losses.append(loss)
        episode_rewards.append(episode_reward)

    agent.save()
    return episode_rewards, episode_losses
