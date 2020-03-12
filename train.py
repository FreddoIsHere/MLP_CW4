import os
import argparse
import numpy as np
from environments import Map_Environment
from ppo_agent import PPO_Agent, train
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='DQN_Agent')
parser.add_argument('--num_episodes', nargs="?", type=int, default=5000, help='number of episodes')
parser.add_argument('--num_particles', nargs="?", type=int, default=2, help='number of particles')
parser.add_argument('--max_steps', nargs="?", type=int, default=50, help='number of steps')
parser.add_argument('--map_file', nargs="?", type=str, default='maps', help='file name')
parser.add_argument('--path_file', nargs="?", type=str, default='paths', help='file name')
parser.add_argument('--path', nargs="?", type=str, default=os.path.abspath(os.getcwd()), help='file name')
args = parser.parse_args()

env = Map_Environment(args.num_particles, args.map_file, args.path_file, np.array([0, 0, 0], dtype=int), np.array([9, 9, 9], dtype=int))
agent = PPO_Agent(env, (1, 10, 10, 10), 12, path=args.path)
rewards, losses = train(env, agent, args.num_episodes, args.max_steps)
plt.plot(rewards)
plt.plot(np.convolve(rewards, (1/50)*np.ones(50), mode='valid'))
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

plt.plot(np.convolve(losses, (1/50)*np.ones(50), mode='valid'))
plt.xlabel("Episode")
plt.ylabel("Average loss")
plt.show()