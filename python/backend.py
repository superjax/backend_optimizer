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
        self.loop_closed = False

class Backend():
    def __init__(self, name="default"):
        self.name = name
        self.G = nx.Graph()
        self.node_plot_positions = dict()
        self.lc_edges = []
        self.LC_threshold = 1.5
        self.overlap_threshold = 0.75
        self.node_id_map = dict()
        self.agents = []
        self.optimized = True

        self.keyframe_index_to_id_map = dict()
        self.keyframe_id_to_index_map = dict()
        self.keyframes = []
        self.current_keyframe_index = 0
        self.optimizer = backend_optimizer.Optimizer()


    def add_keyframe(self, KF, node_id):
        # Add the keyframe to the map
        self.keyframes.append(KF)
        self.keyframe_index_to_id_map[self.current_keyframe_index] = node_id
        self.keyframe_id_to_index_map[node_id] = self.current_keyframe_index
        self.current_keyframe_index += 1


    def add_agent(self, vehicle_id, KF):
        # Tell the backend to keep track of this agent
        new_agent = Agent(vehicle_id)
        self.agents.append(new_agent)
        self.G.add_node(str(vehicle_id)+"_000", KF=KF)

        # Add keyframe to the map
        self.add_keyframe(KF, str(vehicle_id)+"_000")

        if vehicle_id == 0:
            self.agents[0].loop_closed = True


    def add_edge(self, edge):
        # Add this edge to the networkx graph
        # TODO: add checks to make sure we know about this agent
        self.G.add_edge(edge.from_id, edge.to_id, covariance=edge.covariance, transform=edge.transform,
                        from_id=edge.from_id, to_id=edge.to_id)
        # Save the keyframe to the node
        self.G.node[edge.to_id]['KF'] = edge.KF
        #Add Keyframe to the map
        self.add_keyframe(edge.KF, edge.to_id)



    def find_loop_closures(self):
        # Build a KDtree to search
        tree = scipy.spatial.KDTree(self.keyframes)
        lc_count = 0

        print("finding loop closures")
        for from_id in tqdm(self.G.node):
            KF_from = self.G.node[from_id]['KF']
            keyframe_from_index = self.keyframe_id_to_index_map[from_id]
            indices = tree.query_ball_point(KF_from, self.LC_threshold)
            for index in indices:
                if abs(keyframe_from_index - index) > 10:
                    to_id = self.keyframe_index_to_id_map[index]
                    P = [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]
                    KF_to = self.keyframes[index]
                    self.G.add_edge(from_id, to_id, covariance=P,
                                    transform=self.find_transform(np.array(KF_from), np.array(KF_to)),
                                    from_id=from_id, to_id=to_id)
                    lc_count += 1
                    break
        print("found %d loop closures" % lc_count)


    def find_transform(self, from_pose, to_pose):
        # Rotate transform frame to the "from_node" frame
        xij_I = to_pose[0:2] - from_pose[0:2]
        psii = from_pose[2]
        R_I_to_i = np.array([[cos(psii), sin(psii)],
                             [-sin(psii), cos(psii)]])
        dx = xij_I.dot(R_I_to_i.T)

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



    def optimize(self):
        # Find loop closures
        self.find_loop_closures()

        # plt.ion()
        self.plot_graph(self.G, "full graph (TRUTH)", figure_handle=0)
        self.plot_agent(self.G, agent=0, figure_handle=0)
        self.plot_agent(self.G, agent=205, figure_handle=0, color='y')
        self.plot_agent(self.G, agent=750, figure_handle=0, color='g')

        # Get a minimum spanning tree of nodes connected to our origin node
        min_spanning_tree = nx.Graph()
        for component in sorted(nx.connected_component_subgraphs(self.G, copy=True), key=len, reverse=True):
            if '0_000' in component.node:
                self.plot_graph(component, "connected component truth", figure_handle=2, edge_color='r')
                # Find Initial Guess for node positions
                component.node['0_000']['pose'] = [0, 0, 0]
                self.seed_graph(component, '0_000')
                self.plot_graph(component, "connected component unoptimized", figure_handle=3, edge_color='m')
                self.plot_agent(component, agent=0, figure_handle=3)
                self.plot_agent(component, agent=205, figure_handle=0, color='y')
                self.plot_agent(component, agent=750, figure_handle=0, color='g')

                # Let GTSAM crunch it
                print("optimizing")
                optimized_component = self.call_gtsam(component)
                self.plot_graph(optimized_component, "optimized", figure_handle=4, edge_color='b')
                self.plot_agent(optimized_component, agent=0, figure_handle=4)
                self.plot_agent(optimized_component, agent=205, figure_handle=4, color='y')
                self.plot_agent(optimized_component, agent=750, figure_handle=4, color='g')

        plt.show()


    def call_gtsam(self, graph):

        # Build nodes list
        nodes = []
        for i in sorted(graph.nodes_iter()):
            nodes.append([i, graph.node[i]['pose'][0], graph.node[i]['pose'][1], graph.node[i]['pose'][2]])

        # Fix agent 0, node 0 as global origin (Could be moved)
        fixed_node = "0_000"

        # Build edges list
        edges = []
        for pair in graph.edges():
            i = pair[0]
            j = pair[1]
            edge = graph.edge[i][j]
            edges.append([edge['from_id'], edge['to_id'],
                          self.G.edge[i][j]['transform'][0],
                          self.G.edge[i][j]['transform'][1],
                          self.G.edge[i][j]['transform'][2],
                          self.G.edge[i][j]['covariance'][0][0],
                          self.G.edge[i][j]['covariance'][1][1],
                          self.G.edge[i][j]['covariance'][2][2]])

        # run GTSAM
        self.optimizer.new_graph(nodes, edges, fixed_node)
        self.optimizer.optimize()
        optimized_values = self.optimizer.get_optimized()

        # Create a new graph of optimized values
        optimized_graph = nx.Graph()
        for node in optimized_values["nodes"]:
            node_id = node[0]
            optimized_graph.add_node(node_id, pose=[node[1], node[2], node[3]], vehicle_id=node_id.split("_")[0],
                                     KF = self.G.node[node_id]['KF'])

        for edge in optimized_values["edges"]:
            from_id = edge[0]
            to_id = edge[1]
            transform = self.find_transform(np.array(graph.node[from_id]['pose']),
                                            np.array(graph.node[to_id]['pose']))
            # Total guess about covariance for optimized edges
            P = [[0.00001, 0, 0],
                 [0, 0.00001, 0],
                 [0, 0, 0.00001]]
            optimized_graph.add_edge(from_id, to_id, transform=transform, covariance=P)
        return optimized_graph


    def plot_graph(self, graph, title='default', name='default', arrows=False, figure_handle=0, edge_color='m', lc_color='y'):
        if figure_handle:
            plt.figure(figure_handle)
        else:
            plt.figure()
            plt.clf()
        plt.title(title)
        ax = plt.subplot(111)

        # Get positions of all nodes
        plot_positions = dict()
        for (i, n) in graph.node.iteritems():
            if 'pose' in n:
                plot_positions[i] = [n['pose'][1], n['pose'][0]]
            else:
                plot_positions[i] = [n['KF'][1], n['KF'][0]]

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

    def plot_agent(self,graph, agent, figure_handle=0, color='c'):
        nodes = sorted(graph.node)
        agent_nodes = []
        x = []
        y = []
        for n in nodes:
            if int(n.split("_")[0]) == agent:
                if 'pose' in graph.node[n]:
                    x.append(graph.node[n]['pose'][0])
                    y.append(graph.node[n]['pose'][1])
                else:
                    x.append(graph.node[n]['KF'][0])
                    y.append(graph.node[n]['KF'][1])

        if figure_handle:
            plt.figure(figure_handle)
        else:
            plt.figure()

        plt.plot(x, y, color=color, lw=4)


