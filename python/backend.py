#!/usr/bin/env python
# -*- coding: utf-8 -*-
from backend_optimizer import backend_optimizer
import networkx as nx
import matplotlib.pyplot as plt
from math import *
import numpy as np
import regex
from tqdm import tqdm
import subprocess
import scipy.spatial
import sys
import os
# from robot import *
import matplotlib.animation as manimation


class Node():
    def __init__(self, id, pose):
        self.id = id
        self.pose = pose
        self.true_pose = pose

    def set_pose(self, pose):
        self.pose = pose

class Edge():
    def __init__(self, vehicle_id, from_id, to_id, covariance, transform, keyframe):
        self.from_id = from_id
        self.to_id = to_id
        self.covariance = covariance
        self.transform = transform
        self.vehicle_id = vehicle_id
        self.KF = keyframe


class Backend():
    def __init__(self):
        self.LC_threshold = 0.9
        self.graphs = dict()

        self.edge_buffer_for_optimizer = []
        self.lc_buffer_for_optimizer = []
        self.node_buffer_for_optimizer = []

        self.tree = []

        self.keyframe_index_to_id_map = dict()
        self.keyframe_id_to_index_map = dict()
        self.new_keyframes = []
        self.keyframes = []
        self.current_keyframe_index = 0

        self.frame_number = 0

        self.special_agent_id = 0


    def add_agent(self, agent):
        # Initialize and keep track of this agent
        if agent.id in self.graphs:
            debug = 1

        start_node_id = str(agent.id) + "_000"
        
        new_agent = dict()
        new_agent['graph'] = nx.Graph()
        new_agent['connected_agents'] = [agent.id]
        new_agent['node_buffer'] = []
        new_agent['edge_buffer'] = []
        new_agent['lc_buffer'] = []
        new_agent['optimizer'] = backend_optimizer.Optimizer()

        # Add keyframe to the backend
        self.add_keyframe(agent.start_pose, start_node_id)

        # Best Guess of origin for graph
        new_agent['graph'].add_node(start_node_id, pose=[0,0,0], KF=agent.start_pose)
        new_agent['optimizer'].new_graph(start_node_id)

        self.graphs[agent.id] = new_agent


    def add_keyframe(self, KF, node_id):
        self.new_keyframes.append({'KF': KF, 'node_id': node_id})
        self.keyframes.append(KF)
        self.keyframe_index_to_id_map[self.current_keyframe_index] = node_id
        self.keyframe_id_to_index_map[node_id] = self.current_keyframe_index
        self.current_keyframe_index += 1


    def add_odometry(self, edge):

        # Figure out which graph this odometry goes with
        keys = self.graphs.keys()
        added_odom = False
        for id in keys:
            if id in self.graphs and edge.vehicle_id in self.graphs[id]['connected_agents']:
                graph = self.graphs[id]
                graph['graph'].add_edge(edge.from_id, edge.to_id, covariance=edge.covariance,
                                        transform=edge.transform, from_id=edge.from_id, to_id=edge.to_id)
                graph['graph'].node[edge.to_id]['KF'] = edge.KF

                # Add Keyframe to the backend
                self.add_keyframe(edge.KF, edge.to_id)

                try:
                    # Concatenate to get a best guess for the pose of the node
                    from_pose = graph['graph'].node[edge.from_id]['pose']
                    to_pose = self.concatenate_transform(from_pose, edge.transform)
                    graph['graph'].node[edge.to_id]['pose'] = to_pose

                    # Prepare this edge for the optimizer
                    out_node = [edge.to_id, to_pose[0], to_pose[1], to_pose[2]]
                    out_edge = [edge.from_id, edge.to_id, edge.transform[0], edge.transform[1], edge.transform[2],
                                edge.covariance[0][0], edge.covariance[1][1], edge.covariance[2][2]]
                    graph['node_buffer'].append(out_node)
                    graph['edge_buffer'].append(out_edge)

                except KeyError:
                    sys.exit("you tried to concatenate an unconstrained edge")

                self.update_gtsam(id)

        # Find Loop Closures
        self.find_loop_closures()

    def finish_up(self):
        # Find loop closures on this latest batch
        self.find_loop_closures()
        # update optimizer
        self.update_gtsam()


    def update_gtsam(self, id):
        graph = self.graphs[id]
        graph['optimizer'].add_edge_batch(graph['node_buffer'], graph['edge_buffer'])
        graph['optimizer'].add_lc_batch(graph['lc_buffer'])
        graph['lc_buffer'] = []
        graph['node_buffer'] = []
        graph['edge_buffer'] = []
        graph['optimizer'].optimize()

        optimized_values = graph['optimizer'].get_optimized()
        for node in optimized_values["nodes"]:
            node_id = node[0]
            graph['graph'].node[node_id]['pose']=[node[1], node[2], node[3]]


    def find_loop_closures(self):
        # Create a new KDTree with all our keyframes
        self.tree = scipy.spatial.KDTree(self.keyframes)

        # Search for loop closures
        for keyframe in self.new_keyframes:
            keyframe_pose = keyframe['KF']
            from_node_id = keyframe['node_id']
            loop_closures = self.tree.query_ball_point(keyframe_pose, self.LC_threshold)

            # TODO USE THE CLOSEST NODE, not just the first index
            for index in loop_closures:
                to_node_id = self.keyframe_index_to_id_map[index]

                if to_node_id == from_node_id:
                    continue # Don't calculate stupid loop closures

                # Save off vehicle ids
                to_vehicle_id = int(to_node_id.split("_")[0])
                from_vehicle_id = int(from_node_id.split("_")[0])

                # Get loop closure edge transform
                transform = self.find_transform(keyframe_pose, self.keyframes[index])

                # Add noise to loop closure measurements
                covariance = [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]
                noise = np.array([np.random.normal(0, covariance[0][0]),
                                  np.random.normal(0, covariance[1][1]),
                                  np.random.normal(0, covariance[2][2])])
                transform = self.concatenate_transform(transform, noise)

                # Figure out which graphs these agents come from
                from_graph_id = -1
                to_graph_id = -1
                for id, graph in self.graphs.iteritems():
                    if from_vehicle_id in graph['connected_agents']:
                        from_graph_id = id
                    if to_vehicle_id in graph['connected_agents']:
                        to_graph_id = id

                if from_graph_id < 0 or to_graph_id < 0:
                    debug = 1

                # If these are the same graph, then just add a loop closure
                if to_graph_id == from_graph_id:
                    self.graphs[to_graph_id]['graph'].add_edge(from_node_id, to_node_id, covariance=covariance,
                                                               transform=transform, from_id=from_node_id,
                                                               to_id=to_node_id)
                    # Add the loop closure to the GTSAM optimizer
                    opt_edge = [from_node_id, to_node_id, transform[0], transform[1], transform[2],
                                covariance[0][0], covariance[1][1], covariance[2][2]]
                    self.graphs[to_graph_id]['lc_buffer'].append(opt_edge)

                # otherwise we need to merge these two graphs
                else:
                    # Figure out which graph to keep
                    if to_graph_id < from_graph_id:
                        graph1 = to_graph_id
                        graph2 = from_graph_id
                    else:
                        graph1 = from_graph_id
                        graph2 = to_graph_id

                    self.plot()

                    # Merge graph2 into graph1
                    self.merge_graphs(self.graphs[graph1], self.graphs[graph2], to_node_id, from_node_id, transform,
                                      covariance)

                    # Now that we have merged the graphs, we can get rid of graph2
                    del self.graphs[graph2]
                    print "deleting graph ", graph2

                    self.plot()
                    debug = 1

                break # Go on to the next keyframe


    # this function merges graph1 into graph2
    def merge_graphs(self, graph1, graph2, to_node_id, from_node_id, transform, covariance):
        # Figure out the transform between graph origins
        transform_between_graphs = []
        if from_node_id in graph1['graph'].node:
            try:
                transform_to_lc_node = self.concatenate_transform(graph1['graph'].node[from_node_id]['pose'], transform)
                transform_between_graphs = self.concatenate_transform(transform_to_lc_node,
                                                                  self.invert_transform(
                                                                      graph2['graph'].node[to_node_id]['pose']))
            except:
                debug =1
        else:
            # loop closure was "backwards"
            transform_to_lc_node = self.concatenate_transform(graph1['graph'].node[to_node_id]['pose'],
                                                              self.invert_transform(transform))
            transform_between_graphs = self.concatenate_transform(transform_to_lc_node,
                                                                  self.invert_transform(
                                                                      graph2['graph'].node[from_node_id]['pose']))

        # Make a list of new nodes
        new_nodes = [n for n in graph2['graph'].nodes() if n not in graph1['graph'].nodes()]
        new_edges = [e for e in graph2['graph'].edges() if e not in graph1['graph'].edges()]

        # Copy nodes to graph1
        graph1['graph'].add_nodes_from(new_nodes)
        graph1['graph'].add_edges_from(new_edges)

        # Merge the list of connected agents
        graph1['connected_agents'].extend(graph2['connected_agents'])

        # Transform new nodes to the right coordinate frame, and add them to the optimizer
        for node_id in new_nodes:
            try:
                pose = self.concatenate_transform(transform_between_graphs, graph2['graph'].node[node_id]['pose'])
                graph1['graph'].node[node_id]['pose'] = pose
                graph1['graph'].node[node_id]['KF'] = graph2['graph'].node[node_id]['KF']
                graph1['node_buffer'].append([node_id, pose[0], pose[1], pose[2]])
            except:
                debug = 1

        # Add the new edges to the optimizer
        for edge in new_edges:
            edge_map = graph2['graph'].edge[edge[0]][edge[1]]
            graph1['graph'].edge[edge[0]][edge[1]] = edge_map
            try:
                graph1['edge_buffer'].append([edge_map['from_id'], edge_map['to_id'], edge_map['transform'][0],
                                         edge_map['transform'][1], edge_map['transform'][2],
                                         edge_map['covariance'][0][0], edge_map['covariance'][1][1],
                                         edge_map['covariance'][2][2]])
            except:
                debug = 1

        # Add the loop closure to the graph and optimizer
        graph1['graph'].add_edge(from_node_id, to_node_id, covariance=covariance, transform=transform,
                                 from_id=from_node_id, to_id=to_node_id)
        graph1['lc_buffer'].append([from_node_id, to_node_id, transform[0], transform[1], transform[2], covariance[0][0],
                                    covariance[1][1], covariance[2][2]])

        for node in graph1['graph'].nodes():
            if 'pose' not in graph1['graph'].node[node]:
                problem = 1

    # The goal of this function is to calculate the best guess of a transform between myself and this new agent
    def find_transform_to_new_agent(self, agent, from_node, to_node, lc_transform):
        my_pose_after_loop_closure = self.concatenate_transform(self.G.node[from_node]['pose'], lc_transform)
        agent_pose_at_loop_closure = self.agents[agent].robot_ptr.backend.G.node[to_node]['pose']

        other_agent_pose_wrt_me = self.concatenate_transform(my_pose_after_loop_closure,
                                                             self.invert_transform(agent_pose_at_loop_closure))
        return other_agent_pose_wrt_me


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

    def plot(self):
        num_subplots = len(self.graphs)

        rows = int(ceil(num_subplots/num_subplots**0.5))
        cols = int(ceil(num_subplots/float(rows)))

        # Prepare figures
        plt.ion()
        for i in range(2):
            plt.figure(i+1)
            plt.clf()

        i = 0
        names = [' truth', ' optimized']
        truth = [1, 0]
        for id, graph in self.graphs.iteritems():
            i += 1

            for j in range(2):
                plt.figure(j+1)
                ax = plt.subplot(rows, cols, i)
                self.plot_graph(graph['graph'], title=str(id) + names[j], edge_color='g', truth=truth[j],
                                axis_handle=ax)
        plt.pause(0.005) # delay for painting process


        # Save frames for future movie making
        plt.figure(1)
        plt.savefig("movie/truth_" + str(self.frame_number).zfill(4) + ".png")
        plt.figure(2)
        plt.savefig("movie/optimized_" + str(self.frame_number).zfill(4) + ".png")
        self.frame_number += 1
        plt.ioff()


    def find_transform(self, from_pose, to_pose):
        from_pose = np.array(from_pose)
        to_pose = np.array(to_pose)
        # Rotate transform frame to the "from_node" frame
        xij_I = to_pose[0:2] - from_pose[0:2]
        psii = from_pose[2]
        R_i_to_I = np.array([[cos(psii), -sin(psii)],
                             [sin(psii), cos(psii)]])
        dx = xij_I.dot(R_i_to_I)

        if np.shape(dx) != (2,):
            debug  = 1
        dpsi = to_pose[2] - psii

        # Pack up and output the loop closure
        return [dx[0], dx[1], dpsi]

    def plot_graph(self, graph, title='default', name='default', arrows=False, axis_handle=-1, edge_color='m',
                   truth=False, plot_lc=False):
        if axis_handle < 0:
            plt.figure()
            plt.clf()
            axis_handle = plt.subplot(111)
        plt.title(title)

        # Get positions of all nodes
        plot_positions = dict()
        for (i, n) in graph.node.iteritems():
            if truth:
                plot_positions[i] = [n['KF'][1], n['KF'][0]]
            else:
                try:
                    plot_positions[i] = [n['pose'][1], n['pose'][0]]
                except:
                    debug = 1

        plot_graph = graph.copy()

        # Remove Loop Closures
        if not plot_lc:
            for pair in graph.edges():
                i = pair[0]
                j = pair[1]
                vID_i = int(i.split("_")[0])
                vID_j = int(j.split("_")[0])
                if vID_i != vID_j:
                    plot_graph.remove_edge(i, j)

        nx.draw_networkx(plot_graph, pos=plot_positions,
                         with_labels=False, ax=axis_handle, edge_color=edge_color,
                         linewidths="0.3", node_color='c', node_shape='')
        if arrows:
            for i, n in graph.node.iteritems():
                if 'pose' in n:
                    pose = n['pose']
                else:
                    pose = n['KF']
                arrow_length = 1.0
                dx = arrow_length * cos(pose[2])
                dy = arrow_length * sin(pose[2])
                # Be sure to convert to NWU for plotting
                axis_handle.arrow(pose[1], pose[0], dy, dx, head_width=arrow_length*0.15, head_length=arrow_length*0.3, fc='c', ec='b')

        plt.axis("equal")

    def plot_agent(self,graph, agent, axis_handle=-1, color='c', truth=False):
        nodes = sorted(graph.node)
        agent_nodes = []
        x = []
        y = []
        for n in nodes:
            if int(n.split("_")[0]) == agent:
                if truth:
                    x.append(graph.node[n]['KF'][0])
                    y.append(graph.node[n]['KF'][1])
                else:
                    try:
                        x.append(graph.node[n]['pose'][0])
                        y.append(graph.node[n]['pose'][1])
                    except:
                        break


        if axis_handle < 0:
            plt.figure()
            plt.plot(y, x, color=color, lw=4)
        else:
            axis_handle.plot(y, x, color=color, lw=4)









