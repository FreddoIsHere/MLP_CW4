import argparse
import numpy as np
from copy import copy
from ppo_agent import PPO_Agent
from environments import Map_Environment
from mayavi import mlab


class Map_Object:
    def __init__(self, num_particles, map_file, path_file):
        self.num_particles = num_particles
        self.env = Map_Environment(self.num_particles, map_file, path_file, shuffle=True)
        self.data = np.squeeze(np.array(copy(self.env.map)))
        self.optimal_path = np.vstack((self.env.start_pos, self.env.path))
        self.agent = PPO_Agent(self.env, (1, 10, 10, 10), 12)
        if self.data.ndim > 2:
            self.is_3d = True
        else:
            self.is_3d = False
        self.predicted_path = self.predict() + 0.0
        self.optimal_x = self.optimal_path[:, 0] + 0.0
        self.optimal_y = self.optimal_path[:, 1] + 0.0
        if self.is_3d:
            self.optimal_z = self.optimal_path[:, 2] + 0.0

    def predict(self, step_max=50):
        path = [copy(self.env.state)]
        for _ in range(step_max):
            action = self.agent.old_policy.forward(self.env.map)
            _, _, done, _ = self.env.step(action)
            state = copy(self.env.state)
            path.append(state)

            if done:
                break

        return np.array(path)

    def generate_plot(self):
        fig = mlab.figure()
        xx, yy, zz = np.where(self.data == 1)
        mlab.points3d(xx, yy, zz, mode="cube", color=(0, 0, 1), scale_factor=1)
        for p in range(self.num_particles):
            mlab.points3d([self.predicted_path[0, p, 0]], [self.predicted_path[0, p, 1]],
                          [self.predicted_path[0, p, 2]], mode='sphere', color=(0, 1, 0), scale_factor=0.5)
            mlab.plot3d(self.predicted_path[:, p, 0], self.predicted_path[:, p, 1], self.predicted_path[:, p, 2],
                        color=(0, 1, 0))
        mlab.plot3d(self.optimal_x, self.optimal_y, self.optimal_z, color=(1, 0, 0))
        mlab.points3d([self.optimal_x[-1]], [self.optimal_y[-1]], [self.optimal_z[-1]], mode='sphere',
                      color=(1, 0, 0), scale_factor=0.5)
        f = mlab.gcf()
        f.scene.camera.azimuth(16)

        mlab.show()


parser = argparse.ArgumentParser(description='Visualisation')
parser.add_argument('--num_particles', nargs="?", type=int, default=3, help='number of particles')
args = parser.parse_args()
occ = Map_Object(args.num_particles, "maps", "paths")
occ.generate_plot()
