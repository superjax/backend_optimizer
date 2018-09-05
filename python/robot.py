import numpy as np
from math import *
import matplotlib.pyplot as plt
from backend import *

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

        self.start_pose = start_pose

        self.edges = []
        self.true_edges = []
        self.keyframes = []
        self.nodes = [[str(id) + "_000", self.x, self.y, self.psi]]  # For exporting to a file
        self.plt_nodes = [[0, self.x, self.y, self.psi]]
        self.true_nodes = [[0, self.x_true, self.y_true, self.psi_true]]
        self.I_nodes = [[0, self.xI, self.yI, self.psiI]]

    def propagate_dynamics(self, u, dt):
        noise = np.array([[np.random.normal(0, self.G[0, 0])],
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
        keyframe = [self.xI, self.yI, self.psiI]
        self.keyframes.append(keyframe)
        to_id = str(self.id) + "_" + str(self.keyframe_id()).zfill(3)
        from_id = str(self.id) + "_" + str(self.keyframe_id()-1).zfill(3)
        edge = [from_id, to_id, self.x, self.y, self.psi, 1.0/self.G[0, 0], 1.0/self.G[1, 1], 1.0/self.G[2, 2]]
        true_edge = [self.keyframe_id() - 1, self.keyframe_id(), self.x_true, self.y_true, self.psi_true]
        self.edges.append(edge)
        self.true_edges.append(true_edge)

        # Add a node to the list of nodes
        node = self.concatenate_edges(self.nodes[-1], edge)
        self.nodes.append([str(self.id) + "_" + str(len(self.nodes)).zfill(3), node[0], node[1], node[2]])
        self.plt_nodes.append([len(self.plt_nodes), node[0], node[1], node[2]])
        true_node = self.concatenate_edges(self.true_nodes[-1], true_edge)
        self.true_nodes.append([len(self.true_nodes), true_node[0], true_node[1], true_node[2]])
        node_I = self.concatenate_edges(self.I_nodes[-1], true_edge)
        self.I_nodes.append([len(self.I_nodes), node_I[0], node_I[1], node_I[2]])

        # reset state
        self.x = 0
        self.y = 0
        self.psi = 0

        self.x_true = 0
        self.y_true = 0
        self.psi_true = 0

        return edge, keyframe

    def keyframe_id(self):
        return len(self.keyframes)

    def concatenate_edges(self, edge1, edge2):
        # edge1 is the last object in the nodes list
        x0 = edge1[1]
        x1 = edge2[2]

        y0 = edge1[2]
        y1 = edge2[3]

        psi0 = edge1[3]
        psi1 = edge2[4]

        #concatenate edges by rotating second edge into the first edge's frame
        x0 += x1*cos(psi0) - y1*sin(psi0)
        y0 += +x1*sin(psi0) + y1*cos(psi0)
        psi0 += psi1

        return [x0, y0, psi0]

    def draw_trajectory(self):

        plt.figure()
        keyframes = np.array(self.keyframes)
        plt.plot(keyframes[:,1], keyframes[:,0], label="true_edges")
        plt.axis("equal")
        plt.legend()
        plt.show()



















