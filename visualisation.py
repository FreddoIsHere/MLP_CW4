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
from path_generator import Path

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


class Map_Object:
    def __init__(self, data):
        self.data = np.squeeze(np.array(data))
        if self.data.ndim > 2:
            self.is_3d = True
            self.path= Path(data).generate_path()
            self.x = self.path[:,0]
            self.y = self.path[:,1]
            self.z = self.path[:,2]
        else:
            self.is_3d = False
        '''self.x = path3[:,0]
        self.y = path3[:,1]
        self.z = path3[:,2]'''
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
            plt.show()
        else:
            cmap = colors.ListedColormap(['blue', 'white'])
            plt.figure(figsize=(6, 6))
            plt.pcolor(data, cmap=cmap, edgecolors='k', linewidths=1)
            # Start and End markers arbitraily assigned to origin and futherest point
            plt.scatter(-1, -1, s=100, c='g', marker='o')
            plt.scatter(self.data.shape[0], self.data.shape[1], s=100, c='r', marker='o')
        plt.show()


data_provider = DataProvider("maps")
occ = Map_Object(data_provider.get_map())
occ.generate_plot()
