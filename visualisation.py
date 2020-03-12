#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:44:24 2020

@author: Tommy
"""

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from data_providers import DataProvider
from copy import copy
from ppo_agent import PPO_Agent
from environments import Map_Environment
from path_generator import Path

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


class Map_Object:
    def __init__(self, map_file, path_file):
        self.num_particles = 2
        self.env = Map_Environment(self.num_particles, map_file, path_file, np.zeros(3), np.array([9, 9, 9]))
        self.data = np.squeeze(np.array(self.env.map)) > 0
        self.optimal_path = np.vstack(([0, 0, 0], self.env.path))
        self.agent = PPO_Agent(self.env, (1, 10, 10, 10), 12)
        if self.data.ndim > 2:
            self.is_3d = True
        else:
            self.is_3d = False
        self.predicted_path = self.predict() + 0.5
        self.optimal_x = self.optimal_path[:, 0] + 0.5
        self.optimal_y = self.optimal_path[:, 1] + 0.5
        if self.is_3d:
            self.optimal_z = self.optimal_path[:, 2] + 0.5

    def predict(self, step_max=50):
        path = [copy(self.env.state)]
        for _ in range(step_max):
            action = self.agent.old_policy.forward(self.env.map)
            _, _, done, _ = self.env.step(action)
            state = copy(self.env.state)
            path.append(state)

            if done:
                break

        return np.array(path)

    def generate_plot(self):
        occ_grid = self.data
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for p in range(self.num_particles):
            ax.plot([self.predicted_path[0, p, 0]], [self.predicted_path[0, p, 1]], [self.predicted_path[0, p, 2]], markerfacecolor='g', markeredgecolor='k', marker='o', markersize=5, alpha=1.0)
            ax.plot(self.predicted_path[:, p, 0], self.predicted_path[:, p, 1], self.predicted_path[:, p, 2], color='g', linewidth=2)
        ax.plot(self.optimal_x, self.optimal_y, self.optimal_z, color='r', linewidth=3)
        ax.voxels(occ_grid, facecolors='blue', alpha=0.75)
        ax.plot([self.data.shape[0] - 0.5], [self.data.shape[1] - 0.5], [self.data.shape[2] - 0.5], markerfacecolor='r',
                    markeredgecolor='k', marker='o', markersize=5, alpha=1.0)
        plt.show()


occ = Map_Object("maps", "paths")
occ.generate_plot()
