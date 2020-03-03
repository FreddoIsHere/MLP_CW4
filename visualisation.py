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
from ppo_agent import DQN_Agent
from environments import Map_Environment
from path_generator import Path

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


class Map_Object:
    def __init__(self, file):
        data_provider = DataProvider(file)
        data = data_provider.get_map()
        self.data = np.squeeze(np.array(data))
        self.env = Map_Environment(file, np.array([0, 0, 0]), np.array([10, 10, 10]))
        self.agent = DQN_Agent(self.env, (1, 10, 10, 10), 6)
        if self.data.ndim > 2:
            self.is_3d = True
        else:
            self.is_3d = False
        self.path = self.predict(np.array([0, 0, 0]), data) # Path(data).generate_path()
        self.x = self.path[:, 0]
        self.y = self.path[:, 1]
        if self.is_3d:
            self.z = self.path[:, 2]

    def predict(self, state, map, step_max=20):
        path = [state]
        for step in range(step_max):
            action = self.agent.get_action(map, False)
            next_map, reward, done, _ = self.env.step(action)

            map = next_map
            path.append(self.env.state)

            if done:
                break
        return np.array(path)

    def generate_plot(self):
        occ_grid = self.data
        fig = plt.figure()
        if self.is_3d:
            ax = fig.gca(projection='3d')
            ax.voxels(occ_grid, facecolors='blue', edgecolor='k')
            ax.plot([-1], [-1], [-1], markerfacecolor='g', markeredgecolor='k', marker='o', markersize=5, alpha=1.0)
            ax.plot([self.data.shape[0]], [self.data.shape[1]], [self.data.shape[2]], markerfacecolor='r',
                    markeredgecolor='k', marker='o', markersize=5, alpha=1.0)
            ax.plot(self.x,self.y,self.z)
        else:
            cmap = colors.ListedColormap(['white', 'blue'])
            plt.figure(figsize=(6, 6))
            plt.pcolor(self.data, cmap=cmap, edgecolors='k', linewidths=1)
            # Start and End markers arbitraily assigned to origin and futherest point
            plt.scatter(0, 0, s=100, c='g', marker='o')
            plt.scatter(self.data.shape[0], self.data.shape[1], s=100, c='r', marker='o')
            plt.plot(self.x, self.y)
        plt.show()

occ = Map_Object("maps")
occ.generate_plot()
