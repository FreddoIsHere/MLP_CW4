
import numpy as np
import dijkstra3d



class Path:
    def __init__(self, data):
        self.data = 9*np.squeeze(np.array(data)) + 1

    def generate_path(self, method="A*"):
        methods = {"Dijkstra", "A*", "dijkstra"}
        compass = False
        if method == "A*":
            compass = True
        if method not in methods:
            raise TypeError (""+ method + " is not a valid pathing method. Please choose 'Dijkstra' or 'A*'. ")
        self.source = (0, 0, 0)
        self.target = (self.data.shape[0]-1, self.data.shape[1]-1, self.data.shape[2]-1)
        path = dijkstra3d.dijkstra(self.data, self.source, self.target, compass=compass)
            
        return path