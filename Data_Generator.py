import numpy as np
import pandas as pd


class World_Frame_Generator:

    def __init__(self, x_bound, y_bound, file):
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.file = file

    def generate_frame(self, dataset_index):
        frame = np.zeros((self.x_bound, self.y_bound, 1))
        frame[:, 0, 0] = np.ones(self.x_bound)
        frame[:, self.y_bound-1, 0] = np.ones(self.x_bound)
        frame[0, :, 0] = np.ones(self.y_bound)
        frame[self.x_bound-1, :, 0] = np.ones(self.y_bound)
        names = ['x', 'y', 'z']
        ind = pd.MultiIndex.from_product([range(s) for s in frame.shape], names=names)
        df = pd.DataFrame({str(0): frame.flatten()}, index=ind)[str(0)]
        df = df.unstack(level='x').swaplevel().sort_index()
        df.columns = [str(i) for i in range(self.x_bound)]
        df.index.names = ['z', 'y']
        print(df)

world = World_Frame_Generator(10, 10, "hi")
world.generate_frame(2)

