#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:44:24 2020

@author: Tommy
"""

import matplotlib.pyplot as plt
from matplotlib import colors

from mayavi import mlab 

import numpy as np
from data_providers import DataProvider
from ppo_agent import PPO_Agent
from environments import Map_Environment
from path_generator import Path

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


class Map_Object:
    def __init__(self, map_file, path_file):
        data_provider = DataProvider(map_file, path_file)
        data, self.optimal_path = data_provider.get_map()
        for i in range(np.random.randint(low=0, high=100)):
            data, self.optimal_path = data_provider.get_map()
        self.data = np.squeeze(np.array(data))
        self.env = Map_Environment(map_file, path_file, np.array([0, 0, 0]), np.array([9, 9, 9]))
        self.agent = PPO_Agent(self.env, (1, 10, 10, 10), 12)
        if self.data.ndim > 2:
            self.is_3d = True
        else:
            self.is_3d = False
        self.predicted_path = self.predict(np.array([0, 0, 0]), data)  # Path(data).generate_path()
        self.predicted_x = self.predicted_path[:, 0] + 0.5
        self.predicted_y = self.predicted_path[:, 1] + 0.5
        self.optimal_x = self.optimal_path[:, 0] + 0.5
        self.optimal_y = self.optimal_path[:, 1] + 0.5
        if self.is_3d:
            self.predicted_z = self.predicted_path[:, 2] + 0.5
            self.optimal_z = self.optimal_path[:, 2] + 0.5

    def predict(self, state, map, step_max=50):
        path = [state]
        for step in range(step_max):
            action = self.agent.old_policy.forward(map)
            next_map, reward, done, _ = self.env.step(action)

            map = next_map
            path.append(self.env.state)

            if done:
                break
        return np.array(path)

    def generate_plot(self):
        occ_grid = self.data
        
        if self.is_3d:
            fig = mlab.figure()
            xx, yy, zz = np.where(self.data == 1)
            mlab.plot3d(self.predicted_x, self.predicted_y, self.predicted_z, color=(0,1,0))
            mlab.plot3d(self.optimal_x, self.optimal_y, self.optimal_z, color=(1,0,0))
            mlab.points3d(xx, yy, zz, mode="cube", color=(0, 0, 1), scale_factor=1)
            mlab.points3d([0.5], [0.5], [0.5], mode='sphere', color=(0, 1, 0), scale_factor=0.5)
            mlab.points3d([self.data.shape[0] - 0.5], [self.data.shape[1] - 0.5], [self.data.shape[2] - 0.5], mode='sphere', color=(1, 0, 0), scale_factor=0.5)
            mlab.colorbar(title='title', orientation='horizontal', nb_labels=['1', '2'], nb_colors=[(0,0,1), (0,1,0)])

            f = mlab.gcf()
            f.scene.camera.azimuth(16)

            mlab.show()
        else:
            fig = plt.figure()
            cmap = colors.ListedColormap(['white', 'blue'])
            plt.figure(figsize=(6, 6))
            plt.plot(self.predicted_x, self.predicted_y, linewidth=4)
            plt.pcolor(self.data, cmap=cmap, edgecolors='k', linewidths=1)
            # Start and End markers arbitraily assigned to origin and futherest point
            plt.scatter(0, 0, s=100, c='g', marker='o')
            plt.scatter(self.data.shape[0], self.data.shape[1], s=100, c='r', marker='o')
            plt.show()


occ = Map_Object("maps", "paths")
occ.generate_plot()
