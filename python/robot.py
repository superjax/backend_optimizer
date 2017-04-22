import numpy as np
from math import *
import matplotlib.pyplot as plt

class Robot():
    def __init__(self, id, G, start_pose):
        self.id = id
        self.x = 0
        self.x_true = 0
        self.y = 0
        self.y_true = 0
        self.psi = 0
        self.psi_true = 0
        self.G = G

        self.xI = start_pose[0]
        self.yI = start_pose[1]
        self.psiI = start_pose[2]

        self.edges = []
        self.true_edges = []
        self.keyframes = []

    def propagate_dynamics(self, u, dt):
        noise =np.array([[np.random.normal(0, self.G[0, 0])],
                         [np.random.normal(0, self.G[1, 1])],
                         [np.random.normal(0, self.G[2, 2])]])
        v = u[0]
        w = u[1]

        # Calculate Dynamics
        xdot = v * cos(self.psi)
        ydot = v * sin(self.psi)
        psidot = w

        # Euler Integration (noisy)
        self.x +=   (xdot + noise[0,0]) * dt
        self.y +=   (ydot + noise[1,0]) * dt
        self.psi += (psidot + noise[2,0]) * dt
        # wrap psi to +/- PI
        if self.psi > pi:
            self.psi -= 2*pi
        elif self.psi <= -pi:
            self.psi += 2*pi

        # Propagate truth
        self.x_true += xdot * dt
        self.y_true += ydot * dt
        self.psi_true += psidot * dt
        # wrap psi to +/- PI
        if self.psi_true > pi:
            self.psi_true -= 2*pi
        elif self.psi_true <= -pi:
            self.psi_true += 2*pi

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

        return np.array([[self.x, self.y, self.psi]]).T

    def state(self):
        return [self.xI, self.yI, self.psiI]

    def reset(self):
        self.keyframes.append([self.xI, self.yI, self.psiI])
        edge = [self.x, self.y, self.psi]
        true_edge = [self.x_true, self.y_true, self.psi_true]
        self.edges.append(edge)
        self.true_edges.append(true_edge)

        # reset state
        self.x = 0
        self.y = 0
        self.psi = 0

        self.x_true = 0
        self.y_true = 0
        self.psi_true = 0

    def concatenate_edges(self, edge1, edge2):
        x0 = edge1[0]
        x1 = edge2[0]

        y0 = edge1[1]
        y1 = edge2[1]

        psi0 = edge1[2]
        psi1 = edge2[2]

        #concatenate edges by rotating second edge into the first edge's frame
        x0 += x1*cos(psi0) + y1*sin(psi0)
        y0 += -x1*sin(psi0) + y1*cos(psi0)
        psi0 += psi1

        return [x0, y0, psi0]

    def draw_trajectory(self):

        plt.figure()
        keyframes = np.array(self.keyframes)
        plt.plot(keyframes[:,1], keyframes[:,0], label="true_edges")
        plt.axis("equal")
        plt.legend()
        plt.show()



















