from robot import *
import numpy as np

class MC_Robot(Robot):
    def __init__(self, G, start_pose, num_trajectories):
        self.num_trajectories = num_trajectories
        self.x = np.zeros(num_trajectories)
        self.x_true = 0
        self.y = np.zeros(num_trajectories)
        self.y_true = 0
        self.psi = np.zeros(num_trajectories)
        self.psi_true = 0
        self.G = G

        self.xI = start_pose[0]
        self.yI = start_pose[1]
        self.psiI = start_pose[2]

        self.start_pose = start_pose

        self.edges = []
        self.true_edges = []
        self.keyframes = []

    def propagate_dynamics(self, u, dt):
        v = u[0]
        w = u[1]

        # Calculate Dynamics
        xdot = v * cos(self.psi_true)
        ydot = v * sin(self.psi_true)
        psidot = w

        # Propagate truth
        self.x_true += xdot * dt
        self.y_true += ydot * dt
        self.psi_true += psidot * dt
        # wrap psi to +/- PI
        if self.psi_true > pi:
            self.psi_true -= 2*pi
        elif self.psi_true <= -pi:
            self.psi_true += 2*pi

        # Propagate Estimate
        self.x += (xdot*np.ones(self.num_trajectories) + np.random.randn(self.num_trajectories) * self.G[0][0]) * dt
        self.y += (ydot*np.ones(self.num_trajectories) + np.random.randn(self.num_trajectories) * self.G[1][1]) * dt
        self.psi += (psidot*np.ones(self.num_trajectories) + np.random.randn(self.num_trajectories) * self.G[2][2]) * dt

        # Propagate Inertial Truth (for BOW hash)
        xdot = v * cos(self.psiI)
        ydot = v * sin(self.psiI)
        self.xI += xdot * dt
        self.yI += ydot * dt
        self.psiI += w * dt
        # wrap psi to +/- PI
        if self.psiI > pi:
            self.psiI -= 2 * pi
        elif self.psiI <= -pi:
            self.psiI += 2 * pi

        # Wrap to +/0 psi
        out_of_bounds = self.psi > np.pi
        if True in out_of_bounds:
            self.psi[out_of_bounds] -= 2.0*np.pi
        out_of_bounds = self.psi <= -np.pi
        if True in out_of_bounds:
            self.psi[out_of_bounds] += 2.0 * np.pi

        return np.array([self.x, self.y, self.psi])

    def reset(self):
        keyframe = [self.xI, self.yI, self.psiI]
        self.keyframes.append(keyframe)
        edge = np.array([self.x, self.y, self.psi])
        true_edge = [self.x_true, self.y_true, self.psi_true]
        self.edges.append(edge)
        self.true_edges.append(true_edge)

        # reset state
        self.x = np.zeros(self.num_trajectories)
        self.y = np.zeros(self.num_trajectories)
        self.psi = np.zeros(self.num_trajectories)

        self.x_true = 0
        self.y_true = 0
        self.psi_true = 0

        return edge, keyframe

