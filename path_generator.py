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
import dijkstra3d
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


class Path:
    def __init__(self, data):
        self.data = np.squeeze(np.array(data))
        if self.data.ndim > 2:
            self.is_3d = True
        else:
            self.is_3d = False
        print(self.data.shape)

    def generate_path(self, method="A*"):
        self.source = (0,0,0)
        self.target = (self.data.shape[0]-1, self.data.shape[1]-1, self.data.shape[2]-1)
        methods = {"Dijkstra", "A*", "dijkstra"}
        compass = False
        if method == "A*":
            compass = True
        if method not in methods:
            raise TypeError (""+ method + " is not a valid pathing method. Please choose 'Dijkstra' or 'A*'. ")
        path = dijkstra3d.dijkstra(self.data, self.source, self.target, compass=compass) 
        np.append(path,[self.data.shape[0],self.data.shape[1],self.data.shape[2]])
        return path
