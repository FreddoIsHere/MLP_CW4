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
import random
import argparse
import pickle
from tqdm import tqdm
#from visualisation import Map_Object


class Path:
    def __init__(self, data):
        self.data = np.squeeze(np.array(data))
        if self.data.ndim > 2:
            self.is_3d = True
            self.agent_size = (2,2,2)
        else:
            self.is_3d = False
            self.agent_size = (2,2)
        print(self.data.shape)
        

    def generate_path(self, method="A*"):
        methods = {"Dijkstra", "A*", "dijkstra"}
        compass = False
        if method == "A*":
            compass = True
        if method not in methods:
            raise TypeError (""+ method + " is not a valid pathing method. Please choose 'Dijkstra' or 'A*'. ")
        if self.is_3d == True:
            self.source = (0,0,0)
            self.target = (self.data.shape[0]-1, self.data.shape[1]-1, self.data.shape[2]-1)
            self.path = dijkstra3d.dijkstra(self.data, self.source, self.target, compass = compass)
            self.path = np.append(self.path,[[self.data.shape[0],self.data.shape[1],self.data.shape[2]]],axis=0)
        else:
            self.source = (0,0)
            self.target = (self.data.shape[1]-1, self.data.shape[0]-1)
            self.path = dijkstra3d.dijkstra(self.data, self.source, self.target, compass= compass)
            self.path = np.append(self.path,[[self.data.shape[0],self.data.shape[1]]], axis=0)
        #print("walkable path found")
        return np.array(self.path) 

    def make_walkable(self):
        '''Creates a walkable path checking each point is reachable and generates a path can be traversed by agent of specified size'''
        ob_pos=1
        for index,curr_point in enumerate(list(self.path)):
            if not self.walkable(curr_point):
                x = curr_point[0]
                y = curr_point[1]
                if self.is_3d:
                    z = curr_point[2]
                    self.data[x,y,z]=1
                else:
                    self.data[x,y]=1
               # print('path not walkable at point ' + str(index))
                is_walkable=False
                ob_pos=index
                #print(self.path.shape)
                #print(self.data[x,y,z])
                break
            else: 
                is_walkable=True
        if is_walkable == False:
            if ob_pos==0:
                raise StopIteration("No Walkable Path Found")
            #print ('obstacle at: '+str(curr_point))
            self.generate_path()
            walkable_path=self.make_walkable()
            return walkable_path
        else: 
            #print("path is walkable")
            walkable_path = list(self.path)
            return walkable_path
        
    def walkable(self,curr_point,next=None, is_walkable=False):
        a_size = self.agent_size
        if min(self.agent_size)<3:
            if self.is_3d:
                a = np.zeros((3,3,3))
            else:
                a = np.zeros((3,3))
        if (min(curr_point)<max(a_size) )or( max(curr_point)>=max(self.data.shape)-1):
            curr_agent_size = tuple(max(1,a_size[x]//2) for x,y in enumerate(a_size))
            a = np.zeros(curr_agent_size)
            #print('True')
       # else:
           # print(a.shape)
        
        x = curr_point[0]
        y = curr_point[1]
        ax,ay = a.shape[0], a.shape[1]
        
        
        if self.is_3d:
            z = curr_point[2]
            az = a.shape[2]
            slice = self.data[max(0,x-((ax//2))):min(x+((ax//2))+1,self.data.shape[0]),
                          max(0,y-((ay//2))):min(y+((ay//2))+1,self.data.shape[1]),
                          max(0,z-((az//2))):min(z+((az//2))+1, self.data.shape[2])]
        else:
            slice = self.data[max(0,x-((ax//2))):min(x+((ax//2))+1,self.data.shape[0]),
                          max(0,y-((ay//2))):min(y+((ay//2))+1,self.data.shape[1])]
        
        if np.any(a+slice):
            #print(slice)
            return False
            
        else: 
            return True
        
        
    def remove_diags(self, path=None):
        if path==None:
            path = self.path
        
        new_p = np.zeros((1,self.data.ndim))
        p = list(path)
        for index,point in enumerate(p):
            if index<len(p)-1:
                a = p[index]
                b = p[index+1]
                if np.array(np.not_equal(np.array(a),np.array(b))).sum()>1:
                    #print("point " +str(index)+" to "+ str(index+1)+" is diagonal")
                    pos = np.array(np.where(np.not_equal(np.array(b),np.array(a))))
                    #print(pos)
                    newp= point
                    
                    for i in pos.T:
                        print(i)
                       
                        if b[i]>a[i]:
                            newp[i]+=1
                        else:
                            newp[i]-=1
                        newp=np.array(newp)
                        if self.is_3d==True:
                            new_p=np.append(new_p,[[newp[0],newp[1],newp[2]]],axis=0)
                        else:
                            new_p=np.append(new_p,[[newp[0],newp[1]]],axis=0)
                        #print(newp)
                        
                else:
                  #  print("point " +str(index)+" to "+ str(index+1)+" is NOT diagonal")
                    new_p= np.append(new_p, [np.array(point)],axis=0)
       # print(len(new_p))
        
                
        return new_p
                                   
"""data_provider = DataProvider("maps")
path = Path(data_provider.get_map())
x = path.generate_path()
path1 =path.make_walkable() 
p2=path.remove_diags(path1)"""

print('pathing finished')
#pprint(np.array(path1))
#path.smooth()

    
