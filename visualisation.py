#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:44:24 2020

@author: Tommy
"""

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

#Randomly generated grids for test purposes
data = np.random.randint(2,size=(5,5,5))
class Map_Object:
    def __init__(self, data):
        self.data = data 
        if self.data.ndim > 2: 
            self.is_3d = True
        else:
            self.is_3d = False
            
    def generate_plot(self):
        occ_grid = self.data
        fig = plt.figure()
        if self.is_3d: 
            ax = fig.gca(projection='3d')
            ax.voxels(occ_grid, facecolors='blue', edgecolor = 'k')
            ax.plot([0.], [0.], [0.], markerfacecolor='g', markeredgecolor='k', marker='o', markersize=5, alpha=1.0)
            ax.plot([5.], [5.], [5.], markerfacecolor='r', markeredgecolor='k', marker='o', markersize=5, alpha=1.0)
            plt.show()
        else: 
           cmap = colors.ListedColormap(['blue','white'])
           plt.figure(figsize=(6,6))
           plt.pcolor(data,cmap=cmap,edgecolors='k', linewidths=1)
           #Start and End markers arbitraily assigned to origin and futherest point
           plt.scatter(0,0,s=100, c='g', marker='o')
           plt.scatter(5,5, s=100, c='r', marker='o')
        plt.show()
        
occ = Map_Object(data)
occ.generate_plot()

