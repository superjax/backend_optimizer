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
        self.LC_threshold = 0.50
        self.overlap_threshold = 0.75
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
        new_agent['graph'].add_node(start_node_id, pose=[0,0,0])
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
                    if abs(to_graph_id - self.special_agent_id) > abs(from_graph_id - self.special_agent_id):
                        graph1 = from_graph_id
                        graph2 = to_graph_id
                    else:
                        graph2 = to_graph_id
                        graph1 = from_graph_id

                    # Merge graph2 into graph1
                    self.merge_graphs(self.graphs[graph1], self.graphs[graph2], to_node_id, from_node_id, transform,
                                      covariance)

                    # Now that we have merged the graphs, we can get rid of graph2
                    del self.graphs[graph2]
                    print "deleting graph ", graph2

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
                graph1['node_buffer'].append([node_id, pose[0], pose[1], pose[2]])
            except:
                debug = 1

        # Add the new edges to the optimizer
        for edge in new_edges:
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
        dx = -T[0]*cos(T[2]) + T[1]*sin(T[2])
        dy = -T[0]*sin(T[2]) - T[1]*cos(T[2])
        psi = -T[2]
        return [dx, dy, psi]

    def plot(self):
        self.update_gtsam(self.special_agent_id)

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







