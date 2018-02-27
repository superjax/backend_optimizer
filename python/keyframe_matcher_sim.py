#!/usr/bin/env python

from backend_optimizer import kdtree
import numpy as np
from math import *


class KeyframeMatcher():
    def __init__(self, cov):
        self.keyframe_index_to_id_map = dict()
        self.keyframe_id_to_index_map = dict()
        self.new_keyframes = []
        self.keyframes = []
        self.current_keyframe_index = 0
        self.tree = []
        self.keyframe_matcher = kdtree.KDTree()
        self.LC_threshold = 0.9
        self.covariance = cov

    def add_keyframe(self, KF, node_id):
        self.new_keyframes.append([node_id, KF[0], KF[1], KF[2]])

    def find_loop_closures(self):
        loop_cloures = []
        # Create a new KDTree with all our keyframes
        self.keyframe_matcher.add_points(self.new_keyframes)
        self.keyframes.append(self.new_keyframes)

        # Search for loop closures
        for keyframe in self.new_keyframes:
            to_keyframe = self.keyframe_matcher.find_closest_point(keyframe, self.LC_threshold)

            if to_keyframe[0] != u'none':

                new_loop_closure = dict()

                new_loop_closure['to_node_id'] = to_keyframe[0]
                new_loop_closure['from_node_id'] = keyframe[0]

                # Get loop closure edge transform
                transform = self.find_transform(keyframe[1:], to_keyframe[1:])

                # Add noise to loop closure measurements
                noise = np.array([np.random.normal(0, self.covariance[0][0]),
                                  np.random.normal(0, self.covariance[1][1]),
                                  np.random.normal(0, self.covariance[2][2])])
                new_loop_closure['transform'] = self.concatenate_transform(transform, noise)
                new_loop_closure['covariance'] = self.covariance
                loop_cloures.append(new_loop_closure)
        return loop_cloures

    def find_transform(self, from_pose, to_pose):
        from_pose = np.array(from_pose)
        to_pose = np.array(to_pose)
        # Rotate transform frame to the "from_node" frame
        xij_I = to_pose[0:2] - from_pose[0:2]
        psii = from_pose[2]
        R_i_to_I = np.array([[cos(psii), -sin(psii)],
                             [sin(psii), cos(psii)]])
        dx = xij_I.dot(R_i_to_I)
        dpsi = to_pose[2] - psii

        # Pack up and output the loop closure
        return [dx[0], dx[1], dpsi]


    def concatenate_transform(self, T1, T2):
        x = T1[0] + T2[0] * cos(T1[2]) - T2[1] * sin(T1[2])
        y = T1[1] + T2[0] * sin(T1[2]) + T2[1] * cos(T1[2])
        psi = T1[2] + T2[2]
        return [x, y, psi]

    def invert_transform(self, T):
        dx = -(   T[0]*cos(T[2]) + T[1]*sin(T[2]))
        dy = -( - T[0]*sin(T[2]) + T[1]*cos(T[2]))
        psi = -T[2]
        return [dx, dy, psi]
