#! /usr/bin/env python3

import numpy as np
import random


class Space:
    def __init__(self, *args):
        """ Space initializer. It takes the dimension sizes as arguments. Put 2 arguments for 2D, 3 arguments for 3D spaces.
            e.g. Space(5,6,7) means a 5x6x7 space. 

            The space have all zeros at the start. Then ones are inserted representing the obstacles. See add_obstacle function.
        """
        self.__validate_constructor_arguments(args)
        self.array = np.zeros(shape=args, dtype=np.int8)

    def get_array(self):
        """ Returns the space array in numpy.array format.
        
        Returns:
            np.array -- the space
        """
        return self.array

    def add_obstacle(self, obstacle, location):
        """ It adds gıven obstacle to the given location in the space.
        
        Arguments:
            obstacle {Obstacle object} -- The obstacle object which is a multi-dimensional np.array with all ones.
            location {list} -- the start location for adding the obstacle. 2 values for 2 dim, 3 values for 3 dim.
        """
        self.__validate_obstacle_arguments(obstacle, location)
        end_location = location.copy()

        for i, length in enumerate(obstacle.shape):  # find the exact end indices in the space.
            end_location[i] += length

        pad = obstacle.size
        # print(obstacle)
        if len(self.get_array().shape) == 2:
            self.array = np.pad(self.get_array(), (
            (0, pad), (0, pad)))  # 2D - expand the space to prevent index bound errors. It will be undone at the end.
            self.array[location[0]:end_location[0], location[1]:end_location[1]] = obstacle  # 2D - place the obstacle
            self.array = self.array[:-pad, :-pad]  # 2D remove the paddings.

        elif len(self.get_array().shape) == 3:
            self.array = np.pad(self.get_array(), ((0, pad), (0, pad), (
            0, pad)))  # 3D - expand the space to prevent index bound errors. It will be undone at the end.
            self.array[location[0]:end_location[0], location[1]:end_location[1],
            location[2]:end_location[2]] = obstacle  # 3D - place the obstacle
            self.array = self.array[:-pad, :-pad, :-pad]  # 3D remove the paddings.

    def __validate_constructor_arguments(self, args):
        """ It checks if the arguments are in correct format. If not, it throws relevant errors. 
        
        Arguments:
            args {tuple} -- a tuple having integer values. 2 value for 2 dim, 3 value for 3 dim.
        
        Raises:
            ValueError: Minimum 2 arguments needed
            ValueError: Maximum 3 arguments can be defined
            TypeError: All of the arguments should be in integer type.
        """
        if len(args) < 2:
            raise ValueError("Minimum 2 arguments needed.")
        elif len(args) > 3:
            raise ValueError("Maximum 3 arguments can be defined.")
        else:
            for arg in args:
                if not isinstance(arg, int):
                    raise TypeError('"' + str(arg) + '" is not an integer.')

    def __validate_obstacle_arguments(self, obstacle, location):
        """ It checks if the arguments are in correct format. If not, it throws relevant errors. 
        
        Arguments:
            obstacle {Obstacle} -- The obstacle object to be addad to the space.
            location {list} -- the start location for adding the obstacle. 2 values for 2 dim, 3 values for 3 dim.
        
        Raises:
            TypeError: obstacle argument should be in Obstacle type.
            TypeError: location argument should be in list type.
            ValueError: Location should have 2 values for 2 dimension and 3 values for 3 dimention space.
            TypeError: Location values should be in integer type.
        """
        if not isinstance(obstacle, Obstacle):
            raise TypeError('"' + str(obstacle) + '" is not an instance of Obstacle class.')

        if not isinstance(location, list):
            raise TypeError('location: "' + str(location) + '" is not a list.')
        elif len(location) != self.array.ndim:
            raise ValueError(
                'Dimansionality of location ' + str(location) + ' does not match with the space dimension.')
        else:
            for x in location:
                if not isinstance(x, int):
                    raise TypeError('"' + str(x) + '" is not an integer.')


class Obstacle(np.ndarray):
    def __new__(self, dim=2, max_length=4):
        """ The obstacle class as a subclass of np.ndarray. It should be 3 dimensional for 3D space and 2 dimensional for 2D space.
            The size of the obstacle is selected randomly between 0 and the max_length for each dimension.
            All of the values set to one.

        Arguments:
            np {np.ndarray} -- super class
        
        Keyword Arguments:
            dim {int} -- The dimensionality of the obstacle. Options: 2, 3 (default: {2})
            max_length {int} -- Max length of the obstacle. (default: {4})
        
        Returns:
            [np.ndarray] -- the numpy object.
        """
        d = []
        for i in range(dim):
            d.append(random.randint(1, max_length))
        return np.ones(shape=tuple(d), dtype=int).view(Obstacle)


class MapGenerator:
    def __init__(self, *args):
        """ The map generator. It creates a space and allows to user add some obstacles to it.
        """
        self.space = Space(*args)

    def add_obstacle(self, n=1, max_size=4):
        """ This method is used for adding obstacles to the space.
        
        Keyword Arguments:
            n {int} -- number of obstacles (default: {1})
            max_size {int} -- max_size for obstacle creation (default: {4})
        """
        for i in range(n):
            random_location = np.random.randint(0, self.space.get_array().shape[0], self.space.get_array().ndim)
            self.space.add_obstacle(Obstacle(dim=self.space.get_array().ndim, max_length=max_size),
                                    random_location.tolist())

    def get_map(self):
        """ It returns the generated map.
        
        Returns:
            [np.ndarray] -- The map having the obstacles. 
        """
        return self.space.get_array()


generator = MapGenerator(10, 10)
generator.add_obstacle(n=10, max_size=2)

print(generator.get_map())
