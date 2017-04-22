import numpy as np
from math import *
from random import *
from enum import Enum

class Controller:
    def __init__(self, init_pose):
        self.nominal_omega = 5.0
        self.nominal_velocity = 1.0
        self.intersection_radius = .25

        self.grid_size = 10

        psi0 = init_pose[2]
        R = np.array([[cos(psi0), -sin(psi0), 0],
                      [sin(psi0), cos(psi0), 0],
                      [0, 0, 1]])
        self.waypoint = np.array([init_pose]).T + R.dot(np.array([[self.grid_size, 0, 0]]).T)


    def control(self, t, pose):
        # have we entered an intersection?
        delta_x = self.waypoint[0, 0] - pose[0]
        delta_y = self.waypoint[1, 0] - pose[1]
        if abs(delta_x) < self.intersection_radius and abs(delta_y) < self.intersection_radius:
            # We just entered an intersection
            random_number = random()
            if random_number < 0.5:
                # Straight
                next_waypoint = [self.grid_size, 0, 0]
            elif random_number < 0.75:
                # Left
                next_waypoint = [0, -self.grid_size, -pi/2]
            else: # Right
                next_waypoint = [0, self.grid_size, pi/2]
            psi0 = self.waypoint[2, 0]
            R = np.array([[cos(psi0), -sin(psi0), 0],
                          [sin(psi0), cos(psi0), 0],
                          [0, 0, 1]])
            self.waypoint = np.array(self.waypoint) + R.dot(np.array([next_waypoint]).T)
            # Re-evaluate the vector to the goal
            delta_x = self.waypoint[0, 0] - pose[0]
            delta_y = self.waypoint[1, 0] - pose[1]

        # Drive to the next waypoint
        dpsi = atan2(delta_y, delta_x) - pose[2]

        # Wrap to +/- pi
        if dpsi > pi:
            dpsi = atan2(delta_y, delta_x) - 2*pi - pose[2]
        elif dpsi < -pi:
            dpsi = atan2(delta_y, delta_x) + 2*pi - pose[2]


        u = [0, 0]
        u[0] = self.nominal_velocity
        u[1] = dpsi*self.nominal_omega

        return u

    def sat(self, u, umax, umin):
        if u > umax:
            return umax
        elif u < umin:
            return umin
        else:
            return u







