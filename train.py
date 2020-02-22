import argparse
import numpy as np
from environments import Map_Environment
from dqn_agent import DQN_Agent, train

parser = argparse.ArgumentParser(description='DQN_Agent')
parser.add_argument('--num_episodes', nargs="?", type=int, default=100, help='number of episodes')
parser.add_argument('--max_steps', nargs="?", type=int, default=100, help='number of steps')
parser.add_argument('--batch_size', nargs="?", type=int, default=64, help='size of batches')
parser.add_argument('--file', nargs="?", type=str, default='maps', help='file name')
args = parser.parse_args()

env = Map_Environment(args.file, np.array([0, 0, 0]), np.array([50, 50, 50]))
agent = DQN_Agent(env, (50, 50, 50), 3, 6)
rewards = train(env, args.num_episodes, args.max_steps, args.batch_size)