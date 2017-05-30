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

class Agent():
    def __init__(self, robot):
        self.id  = robot.id
        self.loop_closures = []
        self.connected_to_origin = False
        self.robot_ptr = robot

class Backend():
    def __init__(self, agent, origin_KF, name="default"):
        self.name = name
        self.agent_id = agent
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
        self.own_keyframes = []
        self.all_keyframes = []
        self.current_keyframe_index = 0

        self.optimizer = backend_optimizer.Optimizer()

        self.frame_number = 0



    def add_keyframe(self, KF, node_id):
        vid = int(node_id.split("_")[0])
        if vid == self.agent_id:
            self.own_keyframes.append({'node_id': node_id, 'KF': KF})
        self.all_keyframes.append(KF)
        self.keyframe_index_to_id_map[self.current_keyframe_index] = node_id
        self.keyframe_id_to_index_map[node_id] = self.current_keyframe_index
        self.current_keyframe_index += 1


    def add_agent(self, agent):
        # Tell the backend to keep track of this agent
        new_agent = Agent(agent)
        self.agents[agent.id] = new_agent

        # Add keyframe to the map
        self.add_keyframe(agent.start_pose, str(agent.id)+"_000")

        if agent.id == self.agent_id:
            self.G.add_node(str(self.agent_id) + "_000", KF=agent.start_pose)
            self.agents[agent.id].connected_to_origin = True
            self.G.node[str(self.agent_id)+'_000']['pose'] = [0, 0, 0]
            self.optimizer.new_graph(str(self.agent_id)+'_000')
            os.system("mkdir movie/" + str(self.agent_id))


    def add_odometry(self, edge):
        # Add the edge to the networkx Graph
        # Add Keyframe to the backend
        self.add_keyframe(edge.KF, edge.to_id)

        # If the from_node is connected to the graph,
        if edge.from_id in self.G.nodes():
            self.G.add_edge(edge.from_id, edge.to_id, covariance=edge.covariance, transform=edge.transform,
                            from_id=edge.from_id, to_id=edge.to_id)
            self.G.node[edge.to_id]['KF'] = edge.KF

            try:
                # Concatenate to get a best guess for the pose of the node
                from_pose = self.G.node[edge.from_id]['pose']
            except KeyError:
                sys.exit("you tried to concatenate an unconstrained edge")

            psi0 = from_pose[2]
            x = from_pose[0] + edge.transform[0] * cos(psi0) - edge.transform[1] * sin(psi0)
            y = from_pose[1] + edge.transform[0] * sin(psi0) + edge.transform[1] * cos(psi0)
            psi = psi0 + edge.transform[2]
            self.G.node[edge.to_id]['pose'] = [x, y, psi]

            # Prepare this edge for the optimizer
            out_node = [edge.to_id, x, y, psi]
            out_edge = [edge.from_id, edge.to_id, edge.transform[0], edge.transform[1], edge.transform[2],
                        edge.covariance[0][0], edge.covariance[1][1], edge.covariance[2][2]]
            self.node_buffer_for_optimizer.append(out_node)
            self.edge_buffer_for_optimizer.append(out_edge)

            # Find Loop Closures
            self.find_loop_closures()

            # Import new edges from other backends
            for agent_id in self.agents:
                self.import_graph_from_agent(agent_id)

            # Optimize
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

    def import_graph_from_agent(self, agent_id, initializing=False, initial_transform=[0, 0, 0]):
        agent = self.agents[agent_id]
        if agent.connected_to_origin and agent_id != self.agent_id:
            new_nodes = [n for n in agent.robot_ptr.backend.G.nodes() if n not in self.G.nodes()]
            new_edges = [e for e in agent.robot_ptr.backend.G.edges() if e not in self.G.edges()]

            # Transform these new pose estimates into the correct frame

            # First, find transform between origins
            if initializing:
                transform_between_agents = initial_transform
            else:
                transform_between_agents = self.G.node[str(agent_id)+"_000"]['pose']

            # Add new nodes to the graph
            self.G.add_nodes_from(new_nodes)
            self.G.add_edges_from(new_edges)

            # Apply this transform to all new node pose estimates (edges are relative, so we don't need to do anything to them)
            for node_id in new_nodes:
                self.G.node[node]['pose'] = self.concatenate_transform(transform_between_agents, self.G.node[node]['pose'])



            # Add nodes and edges to the optimizer
            for node in new_nodes:
                self.node_buffer_for_optimizer.append([node['id'], node['pose'][0], node['pose'][1], node['pose'][2]])
            for edge in new_edges:
                self.edge_buffer_for_optimizer.append([edge['from_node_id'], edge['to_node_id'], edge['transform'][0],
                                                       edge['transform'][1], edge['transform'][2],
                                                       edge['covariance'][0][0], edge['covariance'][1][1],
                                                       edge['covariance'][2][2]])


    def find_loop_closures(self):
        self.tree = scipy.spatial.KDTree(self.all_keyframes)
        for keyframe in self.own_keyframes:
            keyframe_pose = keyframe['KF']
            from_node_id = keyframe['node_id']
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
                try:
                    transform = self.find_transform(keyframe_pose, self.all_keyframes[index])
                except:
                    debug = 1

                # Add noise to loop closure measurements
                covariance = [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]
                noise = np.array([np.random.normal(0, covariance[0][0]),
                                  np.random.normal(0, covariance[1][1]),
                                  np.random.normal(0, covariance[2][2])])
                transform = self.concatenate_transform(transform, noise)

                # If only one of the agents have been connected to the origin
                if not self.agents[to_vehicle_id].connected_to_origin:
                    self.agents[to_vehicle_id].connected_to_origin = True
                    other_agent_transform = self.find_transform_to_new_agent(to_vehicle_id, from_node_id, to_node_id, transform)
                    self.import_graph_from_agent(to_vehicle_id, initializing=True, initial_transform=other_agent_transform)

                    # Add the edge to the networkx graph
                    self.G.add_edge(from_node_id, to_node_id, covariance=covariance, transform=transform,
                                    from_id=from_node_id, to_id=to_node_id)

                    # Add the edge to the GTSAM optimizer
                    opt_edge = [from_node_id, to_node_id, transform[0], transform[1], transform[2],
                                covariance[0][0], covariance[1][1], covariance[2][2]]
                    self.lc_buffer_for_optimizer.append(opt_edge)
                    # Run the optimizer (adding a new agent usually means we just added a ton of stuff to the graph)
                    self.update_gtsam()
                    break

                # If both agents are already connected to the origin
                elif self.agents[to_vehicle_id].connected_to_origin and self.agents[from_vehicle_id].connected_to_origin:
                    # Add the edge to the networkx graph
                    self.G.add_edge(from_node_id, to_node_id, covariance=covariance, transform=transform,
                                    from_id=from_node_id, to_id=to_node_id)

                    # Add the edge to the GTSAM optimizer
                    opt_edge = [from_node_id, to_node_id, transform[0], transform[1], transform[2],
                                covariance[0][0], covariance[1][1], covariance[2][2]]
                    self.lc_buffer_for_optimizer.append(opt_edge)
                    break
                else:
                    print("problem")

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
        dx = -T[0]*cos(T[2]) + T[1]*sin(T[2])
        dy = -T[0]*sin(T[2]) - T[1]*cos(T[2])
        psi = -T[2]
        return [dx, dy, psi]

    def plot(self):
        self.update_gtsam()

        plt.ion()
        for i in range(3):
            plt.figure(i+1)
            plt.clf()

        self.plot_graph(self.G, title=str(self.agent_id)+"truth", edge_color='g', truth=True, figure_handle=self.agent_id*10 + 1)
        self.plot_graph(self.G, title=str(self.agent_id)+"unoptimized", edge_color='r', figure_handle=self.agent_id*10 + 2)
        self.plot_agent(self.G, agent=0, figure_handle=self.agent_id*10 + 1, color='c', truth=True)
        self.plot_agent(self.G, agent=0, figure_handle=self.agent_id*10 + 2, color='c')

        self.plot_agent(self.G, agent=1, figure_handle=self.agent_id*10 + 1, color='y', truth=True)
        self.plot_agent(self.G, agent=1, figure_handle=self.agent_id*10 + 2, color='y')

        self.plot_agent(self.G, agent=2, figure_handle=self.agent_id*10 + 1, color='m', truth=True)
        self.plot_agent(self.G, agent=2, figure_handle=self.agent_id*10 + 2, color='m')

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

        self.plot_graph(optimized_graph, title=str(self.agent_id)+"optimized", edge_color='b', figure_handle=self.agent_id*10 + 3)
        self.plot_agent(optimized_graph, agent=0, figure_handle=self.agent_id*10 + 3, color='c')
        self.plot_agent(optimized_graph, agent=1, figure_handle=self.agent_id*10 + 3, color='y')
        self.plot_agent(optimized_graph, agent=2, figure_handle=self.agent_id*10 + 3, color='m')
        plt.pause(0.005) # delay for painting process


        # Save frames for future movie making
        plt.figure(1)
        plt.savefig("movie/"+str(self.agent_id)+"/truth_" + str(self.frame_number).zfill(4) + ".png")
        plt.figure(2)
        plt.savefig("movie/"+str(self.agent_id)+"/unoptimized_" + str(self.frame_number).zfill(4) + ".png")
        plt.figure(3)
        plt.savefig("movie/"+str(self.agent_id)+"/optimized_" + str(self.frame_number).zfill(4) + ".png")
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
            if str(self.agent_id)+'_000' in subcomponent.nodes():
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






