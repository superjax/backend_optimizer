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

class Agent():
    def __init__(self, id):
        self.id  = id
        self.loop_closures = []
        self.connected_to_origin = False

class Backend():
    def __init__(self, agent, origin_KF, name="default"):
        self.name = name
        self.agent = agent
        self.G = nx.Graph()
        self.LC_threshold = 0.50
        self.overlap_threshold = 0.75
        self.node_id_map = dict()
        self.agents = dict()
        self.optimized = True

        self.odometry_buffer = []
        self.edge_buffer_for_optimizer = []
        self.lc_buffer_for_optimizer = []
        self.node_buffer_for_optimizer = []
        self.buffer_size = 0
        self.tree = []
        self.new_keyframes = []
        self.loop_closure_throttle_counter = 0
        self.number_of_loop_closed_agents = 0

        self.keyframe_index_to_id_map = dict()
        self.keyframe_id_to_index_map = dict()
        self.keyframes = []
        self.current_keyframe_index = 0

        self.optimizer = backend_optimizer.Optimizer()

        self.frame_number = 0



    def add_keyframe(self, KF, node_id):
        # Add the keyframe to the map
        self.new_keyframes.append(node_id)
        self.keyframes.append(KF)
        self.keyframe_index_to_id_map[self.current_keyframe_index] = node_id
        self.keyframe_id_to_index_map[node_id] = self.current_keyframe_index
        self.current_keyframe_index += 1


    def add_agent(self, vehicle_id, KF):
        # Tell the backend to keep track of this agent
        new_agent = Agent(vehicle_id)
        self.agents[vehicle_id] = new_agent
        self.G.add_node(str(vehicle_id)+"_000", KF=KF)

        # Add keyframe to the map
        self.add_keyframe(KF, str(vehicle_id)+"_000")

        if vehicle_id == self.agent:
            self.agents[vehicle_id].connected_to_origin = True
            self.G.node[str(self.agent)+'_000']['pose'] = [0, 0, 0]
            self.optimizer.new_graph(str(self.agent)+'_000')
            os.system("mkdir movie/" + str(self.agent))


    def add_odometry(self, edge):
        # Add the edge to the networkx Graph
        self.G.add_edge(edge.from_id, edge.to_id, covariance=edge.covariance, transform=edge.transform,
                        from_id=edge.from_id, to_id=edge.to_id)
        self.G.node[edge.to_id]['KF'] = edge.KF

        #Add Keyframe to the backend
        self.add_keyframe(edge.KF, edge.to_id)

        # Prepare this edge for Optimizer
        vehicle_id = int(edge.from_id.split("_")[0])

        if self.agents[vehicle_id].connected_to_origin:
            # Guess on pose of this node
            try:
                from_pose = self.G.node[edge.from_id]['pose']
                psi0 = from_pose[2]
                x = from_pose[0] + edge.transform[0] * cos(psi0) - edge.transform[1] * sin(psi0)
                y = from_pose[1] + edge.transform[0] * sin(psi0) + edge.transform[1] * cos(psi0)
                psi = psi0 + edge.transform[2]
                self.G.node[edge.to_id]['pose'] = [x, y, psi]
            except KeyError:
                sys.exit("you tried to concatenate an unconstrained edge")

            out_node = [edge.to_id, x, y, psi]
            out_edge = [edge.from_id, edge.to_id, edge.transform[0], edge.transform[1], edge.transform[2],
                        edge.covariance[0][0], edge.covariance[1][1], edge.covariance[2][2]]
            self.node_buffer_for_optimizer.append(out_node)
            self.edge_buffer_for_optimizer.append(out_edge)

        # Throttle update
        self.loop_closure_throttle_counter += 1
        if self.loop_closure_throttle_counter >= 500:
            self.loop_closure_throttle_counter = 0
            self.find_loop_closures()


        if len(self.node_buffer_for_optimizer) >= 2500:
            self.update_gtsam()


    def finish_up(self):
        # Find loop closures on this latest batch
        self.find_loop_closures()
        # update optimizer
        self.update_gtsam()

    def update_gtsam(self):
        self.optimizer.add_edge_batch(self.node_buffer_for_optimizer, self.edge_buffer_for_optimizer)

        self.node_buffer_for_optimizer = []
        self.edge_buffer_for_optimizer = []


        self.optimizer.add_lc_batch(self.lc_buffer_for_optimizer)

        self.lc_buffer_for_optimizer = []
        self.optimizer.optimize()
        self.optimizer.optimize()


    def update_keyframe_tree(self):
        self.tree = scipy.spatial.KDTree(self.keyframes)


    def find_loop_closures(self):
        self.tree = scipy.spatial.KDTree(self.keyframes)
        for from_node_id in self.new_keyframes:
            keyframe_pose = self.G.node[from_node_id]['KF']
            indices = self.tree.query_ball_point(keyframe_pose, self.LC_threshold)

            if len(self.lc_buffer_for_optimizer) > 500:
                self.update_gtsam()

            # TODO USE THE CLOSEST NODE, not just the first index
            for index in indices:
                to_node_id = self.keyframe_index_to_id_map[index]

                if to_node_id == from_node_id:
                    continue # Don't calculate stupid loop closures

                to_vehicle_id = int(to_node_id.split("_")[0])
                from_vehicle_id = int(from_node_id.split("_")[0])

                # Get loop closure edge transform
                transform = self.find_transform(keyframe_pose, self.G.node[to_node_id]['KF'])

                # Add the edge to the networkx graph
                covariance = [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]
                self.G.add_edge(from_node_id, to_node_id, covariance=covariance, transform=transform,
                                from_id=from_node_id, to_id=to_node_id)

                # If only one of the agents have been connected to the origin
                if self.agents[to_vehicle_id].connected_to_origin != self.agents[from_vehicle_id].connected_to_origin:
                    if not self.agents[to_vehicle_id].connected_to_origin:
                        self.initialize_new_loop_closed_agent(to_vehicle_id)
                    else:
                        self.initialize_new_loop_closed_agent(from_vehicle_id)
                    # Add the edge to the GTSAM optimizer
                    opt_edge = [from_node_id, to_node_id, transform[0], transform[1], transform[2],
                                covariance[0][0], covariance[1][1], covariance[2][2]]
                    self.lc_buffer_for_optimizer.append(opt_edge)
                    self.update_gtsam()
                    self.number_of_loop_closed_agents += 1
                    break

                # If both agents are already connected to the origin
                if self.agents[to_vehicle_id].connected_to_origin and self.agents[from_vehicle_id].connected_to_origin:
                    # Add the edge to the GTSAM optimizer
                    opt_edge = [from_node_id, to_node_id, transform[0], transform[1], transform[2],
                                covariance[0][0], covariance[1][1], covariance[2][2]]
                    self.lc_buffer_for_optimizer.append(opt_edge)
                    break

                # Neither agent has been connected to the origin
                else:
                    # Register that the vehicles are loop closed
                    if from_vehicle_id not in self.agents[to_vehicle_id].loop_closures:
                        self.agents[to_vehicle_id].loop_closures.append(from_vehicle_id)
                    if to_vehicle_id not in self.agents[from_vehicle_id].loop_closures:
                        self.agents[from_vehicle_id].loop_closures.append(to_vehicle_id)
                    break
        self.new_keyframes = []


    def initialize_new_loop_closed_agent(self, agent):
        # Initialize all the poses to the beginning of the agent
        path_to_start_of_agent = nx.shortest_path(self.G, str(self.agent)+"_000", str(agent)+"_000")

        for i in range(len(path_to_start_of_agent)):
            node = path_to_start_of_agent[i]
            if not 'pose' in self.G.node[node]:
                edge = self.G.edge[path_to_start_of_agent[i-1]][node]

                if edge['from_id'] == path_to_start_of_agent[i-1]:
                    from_pose = self.G.node[path_to_start_of_agent[i-1]]['pose']
                    psi0 = from_pose[2]
                    x = from_pose[0] + edge['transform'][0] * cos(psi0) - edge['transform'][1] * sin(psi0)
                    y = from_pose[1] + edge['transform'][0] * sin(psi0) + edge['transform'][1] * cos(psi0)
                    psi = psi0 + edge['transform'][2]
                else:
                    to_pose = self.G.node[path_to_start_of_agent[i - 1]]['pose']
                    psi = to_pose[2] - edge['transform'][2]
                    x = to_pose[0] - edge['transform'][0] * cos(psi) + edge['transform'][1] * sin(psi)
                    y = to_pose[1] - edge['transform'][0] * sin(psi) - edge['transform'][1] * cos(psi)
                # Save off the new pose for the networkx graph
                self.G.node[node]['pose'] = [x, y, psi]

                # Also pump it into the optimizer
                out_node = [node, x, y, psi]
                out_edge = [edge['from_id'], edge['to_id'], edge['transform'][0], edge['transform'][1], edge['transform'][2],
                            edge['covariance'][0][0], edge['covariance'][1][1], edge['covariance'][2][2]]
                self.node_buffer_for_optimizer.append(out_node)
                self.edge_buffer_for_optimizer.append(out_edge)

        # Initialize all the other poses
        last_node = self.find_last_node_for_agent(agent)
        path_to_last_node = [str(agent)+"_"+ str(i).zfill(3) for i in range(last_node +1 )]

        for i in range(len(path_to_last_node)):
            node = path_to_last_node[i]
            if not 'pose' in self.G.node[node]:
                edge = self.G.edge[path_to_last_node[i-1]][node]
                if edge['from_id'] == path_to_last_node[i-1]:
                    from_pose = self.G.node[path_to_last_node[i-1]]['pose']
                    psi0 = from_pose[2]
                    x = from_pose[0] + edge['transform'][0] * cos(psi0) - edge['transform'][1] * sin(psi0)
                    y = from_pose[1] + edge['transform'][0] * sin(psi0) + edge['transform'][1] * cos(psi0)
                    psi = psi0 + edge['transform'][2]
                else:
                    to_pose = self.G.node[path_to_last_node[i - 1]]['pose']
                    psi = to_pose[2] - edge['transform'][2]
                    x = to_pose[0] - edge['transform'][0] * cos(psi) + edge['transform'][1] * sin(psi)
                    y = to_pose[1] - edge['transform'][0] * sin(psi) - edge['transform'][1] * cos(psi)
                self.G.node[node]['pose'] = [x, y, psi]

                # Pump these into the optimizer as well
                out_node = [node, x, y, psi]
                out_edge = [edge['from_id'], edge['to_id'], edge['transform'][0], edge['transform'][1], edge['transform'][2],
                            edge['covariance'][0][0], edge['covariance'][1][1], edge['covariance'][2][2]]
                self.node_buffer_for_optimizer.append(out_node)
                self.edge_buffer_for_optimizer.append(out_edge)
        # signal that this agent has been loop-closed
        self.agents[agent].connected_to_origin = True

        # Signal that we need to do an update
        self.loop_closure_throttle_counter = 10000

        # Recursively initialize any agents connected to this agent
        for connected_agent in self.agents[agent].loop_closures:
            if not self.agents[connected_agent].connected_to_origin:
                self.initialize_new_loop_closed_agent(connected_agent)



    def find_last_node_for_agent(self, agent):
        max_node = 0
        for n in self.G.nodes():
            vid = int(n.split("_")[0])
            if vid == agent:
                nnum = int(n.split("_")[1])
                if nnum > max_node:
                    max_node = nnum
        return max_node



    def plot(self):
        self.update_gtsam()

        plt.ion()
        for i in range(3):
            plt.figure(i+1)
            plt.clf()

        self.plot_graph(self.G, title=str(self.agent)+"truth", edge_color='g', truth=True, figure_handle=self.agent*10 + 1)
        self.plot_graph(self.G, title=str(self.agent)+"unoptimized", edge_color='r', figure_handle=self.agent*10 + 2)
        self.plot_agent(self.G, agent=0, figure_handle=self.agent*10 + 1, color='c', truth=True)
        self.plot_agent(self.G, agent=0, figure_handle=self.agent*10 + 2, color='c')

        self.plot_agent(self.G, agent=1, figure_handle=self.agent*10 + 1, color='y', truth=True)
        self.plot_agent(self.G, agent=1, figure_handle=self.agent*10 + 2, color='y')

        self.plot_agent(self.G, agent=2, figure_handle=self.agent*10 + 1, color='m', truth=True)
        self.plot_agent(self.G, agent=2, figure_handle=self.agent*10 + 2, color='m')

        # Create a new graph of optimized values
        optimized_values = self.optimizer.get_optimized()
        optimized_graph = nx.Graph()
        for node in optimized_values["nodes"]:
            node_id = node[0]
            optimized_graph.add_node(node_id,
                                     pose=[node[1], node[2], node[3]],
                                     vehicle_id=node_id.split("_")[0],
                                     KF=self.G.node[node_id]['KF'])

        for edge in optimized_values["edges"]:
            from_id = edge[0]
            to_id = edge[1]
            transform = [edge[2], edge[3], edge[4]]
            # Total guess about covariance for optimized edges
            P = [edge[5], 0, 0, 0, edge[6], 0, 0, 0, edge[7]]
            optimized_graph.add_edge(from_id, to_id, transform=transform, covariance=P)

        self.plot_graph(optimized_graph, title=str(self.agent)+"optimized", edge_color='b', figure_handle=self.agent*10 + 3)
        self.plot_agent(optimized_graph, agent=0, figure_handle=self.agent*10 + 3, color='c')
        self.plot_agent(optimized_graph, agent=1, figure_handle=self.agent*10 + 3, color='y')
        self.plot_agent(optimized_graph, agent=2, figure_handle=self.agent*10 + 3, color='m')
        plt.pause(0.005) # delay for painting process


        # Save frames for future movie making
        plt.figure(1)
        plt.savefig("movie/"+str(self.agent)+"/truth_" + str(self.frame_number).zfill(4) + ".png")
        plt.figure(2)
        plt.savefig("movie/"+str(self.agent)+"/unoptimized_" + str(self.frame_number).zfill(4) + ".png")
        plt.figure(3)
        plt.savefig("movie/"+str(self.agent)+"/optimized_" + str(self.frame_number).zfill(4) + ".png")
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

    def find_pose(self, graph, node, origin_node):
        if 'pose' in graph.node[node]:
            return graph.node[node]['pose']
        else:
            path_to_origin = nx.shortest_path(graph, node, origin_node)
            edge = graph.edge[node][path_to_origin[1]]
            # find the pose of the next closest node (this is recursive)
            nearest_known_pose = self.find_pose(graph, path_to_origin[1], origin_node)

            # edges could be going in either direction
            # if the edge is pointing to this node
            if edge['from_id'] == path_to_origin[1]:
                psi0 = nearest_known_pose[2]
                x = nearest_known_pose[0] + edge['transform'][0] * cos(psi0) - edge['transform'][1] * sin(psi0)
                y = nearest_known_pose[1] + edge['transform'][0] * sin(psi0) + edge['transform'][1] * cos(psi0)
                psi = psi0 + edge['transform'][2]
            else:
                psi = nearest_known_pose[2] - edge['transform'][2]
                x = nearest_known_pose[0] - edge['transform'][0] * cos(psi) + edge['transform'][1] * sin(psi)
                y = nearest_known_pose[1] - edge['transform'][0] * sin(psi) - edge['transform'][1] * cos(psi)

            graph.node[node]['pose'] = [x, y, psi]

            return [x, y, psi]

    def seed_graph(self, graph, node):
        children = [node]
        more_children = True
        depth = 0

        while more_children:
            depth += 1
            more_children = False
            next_iteration_children = []
            for child in children:
                this_child_children, this_node_has_more_children = self.get_children_pose(graph, child)
                if this_node_has_more_children:
                    more_children = True
                    next_iteration_children.extend(this_child_children)

            children = next_iteration_children
        print("network depth = %d", depth)


    def get_children_pose(self, graph, node):
        more_children = False
        children = []
        for child in graph.neighbors(node):
            if 'pose' not in graph.node[child]:
                more_children = True
                children.append(child)
                edge = graph.edge[node][child]

                if edge['from_id'] == node: # edge is pointing from node to child
                    psi0 = graph.node[node]['pose'][2]
                    x = graph.node[node]['pose'][0] + edge['transform'][0] * cos(psi0) - edge['transform'][1] * sin(psi0)
                    y = graph.node[node]['pose'][1] + edge['transform'][0] * sin(psi0) + edge['transform'][1] * cos(psi0)
                    psi = psi0 + edge['transform'][2]
                else: # edge is pointing from child to node
                    psi = graph.node[node]['pose'][2] - edge['transform'][2]
                    x = graph.node[node]['pose'][0] - edge['transform'][0] * cos(psi) + edge['transform'][1] * sin(psi)
                    y = graph.node[node]['pose'][1] - edge['transform'][0] * sin(psi) - edge['transform'][1] * cos(psi)

                graph.node[child]['pose'] = [x, y, psi]
        return children, more_children


    def plot_graph(self, graph, title='default', name='default', arrows=False, figure_handle=0, edge_color='m',
                   truth=False, plot_lc=False):
        if figure_handle:
            plt.figure(figure_handle)
        else:
            plt.figure()
            plt.clf()
        plt.title(title)
        ax = plt.subplot(111)

        # If we are trying to plot estimates, only plot the connected component
        for subcomponent in nx.connected_component_subgraphs(graph):
            if str(self.agent)+'_000' in subcomponent.nodes():
                graph = subcomponent.copy()

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

        # Remove Loop Closures
        if not plot_lc:
            for pair in graph.edges():
                i = pair[0]
                j = pair[1]
                vID_i = int(i.split("_")[0])
                vID_j = int(j.split("_")[0])
                if vID_i != vID_j:
                    graph.remove_edge(i, j)

        nx.draw_networkx(graph, pos=plot_positions,
                         with_labels=False, ax=ax, edge_color=edge_color,
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
                ax.arrow(pose[1], pose[0], dy, dx, head_width=arrow_length*0.15, head_length=arrow_length*0.3, fc='c', ec='b')

        # # Plot loop closures
        # for lc in self.lc_edges:
        #     x = [lc[0][0], lc[1][0]]
        #     y = [lc[0][1], lc[1][1]]
        #     self.ax.plot(y, x , lc_color, lw='0.1', zorder=1)

        plt.axis("equal")

    def plot_agent(self,graph, agent, figure_handle=0, color='c', truth=False):
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


        if figure_handle:
            plt.figure(figure_handle)
        else:
            plt.figure()

        plt.plot(y, x, color=color, lw=4)






