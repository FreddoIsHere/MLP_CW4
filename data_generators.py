import numpy as np


class World_Frame_Generator:

    def __init__(self, file, x_dim, y_dim, z_dim=1):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.file = file

    def generate_frame(self, num_of_frames):
        data = []
        if self.z_dim < 2:
            for _ in range(num_of_frames):
                frame = np.zeros((self.x_dim, self.y_dim))
                frame[:, 0] = np.ones(self.x_dim)
                frame[0, :] = np.ones(self.y_dim)
                frame[self.x_dim-1, :] = np.ones(self.y_dim)
                frame[:, self.y_dim-1] = np.ones(self.x_dim)
                data.append(frame)
        else:
            for _ in range(num_of_frames):
                frame = np.zeros((self.x_dim, self.y_dim, self.z_dim))
                frame[:, :, 0] = np.ones((self.x_dim, self.y_dim))
                frame[:, 0, :] = np.ones((self.x_dim, self.z_dim))
                frame[0, :, :] = np.ones((self.y_dim, self.z_dim))
                frame[self.x_dim - 1, :, :] = np.ones((self.y_dim, self.z_dim))
                frame[:, self.y_dim - 1, :] = np.ones((self.x_dim, self.z_dim))
                frame[:, :, self.z_dim - 1] = np.ones((self.x_dim, self.y_dim))
                data.append(frame)
        np.savez(self.file, inputs=data)


world = World_Frame_Generator("data.npz", 10, 10)
world.generate_frame(2)
out = np.load('data.npz')
print(out['inputs'])
