from collections import deque
import random
import numpy as np


class Memory:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, map, action, reward, log_prob, done):
        experience = (map, action, reward, log_prob, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        map_batch = []
        action_batch = []
        reward_batch = []
        log_prob_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            map, action, reward, log_prob, done = experience
            map_batch.append(map)
            action_batch.append(action)
            reward_batch.append(reward)
            log_prob_batch.append(log_prob)
            done_batch.append(done)

        return map_batch, action_batch, reward_batch, log_prob_batch, done_batch

    def __len__(self):
        return len(self.buffer)
