import os
import argparse
import numpy as np
from environments import Map_Environment
from dqn_agent import DQN_Agent, train
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='DQN_Agent')
parser.add_argument('--num_episodes', nargs="?", type=int, default=200, help='number of episodes')
parser.add_argument('--max_steps', nargs="?", type=int, default=20, help='number of steps')
parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='size of batches')
parser.add_argument('--file', nargs="?", type=str, default='maps', help='file name')
parser.add_argument('--path', nargs="?", type=str, default=os.path.abspath(os.getcwd()), help='file name')
args = parser.parse_args()

env = Map_Environment(args.file, np.array([0, 0, 0]), np.array([9, 9, 9]))
agent = DQN_Agent(env, (1, 10, 10, 10), 6, path=args.path)
rewards, losses = train(env, agent, args.num_episodes, args.max_steps, args.batch_size)
plt.plot(rewards)
plt.plot(np.convolve(rewards, (1/10)*np.ones(10)))
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

plt.plot(losses)
plt.plot(np.convolve(rewards, (1/10)*np.ones(10)))
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.show()