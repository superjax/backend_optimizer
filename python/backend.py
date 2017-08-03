#!/usr/bin/env python
# -*- coding: utf-8 -*-
from backend_optimizer import backend_optimizer
from backend_optimizer import kdtree
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from math import *
import numpy as np
import pickle
from operator import itemgetter


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

        self.keyframe_matcher = kdtree.KDTree()

        self.optimization_counter = 0
        self.num_agents = 0

        self.average_optimization_time_count = 0
        self.average_optimization_time_sum = 0
        self.max_optimization_time = 0
        self.optimization_time_array = []

        # Initialize Plot sizing
        font = {'family': 'normal',
                'weight': 'normal',
                'size': 8}

        matplotlib.use('GTKAgg')
        matplotlib.rc('font', **font)

        self.init_plots(4)


    def add_agent(self, agent):
        # Initialize and keep track of this agent
        start_node_id = str(agent.id) + "_000"
        
        new_agent = dict()
        new_agent['graph'] = nx.Graph()
        new_agent['connected_agents'] = [agent.id]
        new_agent['node_buffer'] = []
        new_agent['edge_buffer'] = []
        new_agent['optimizer'] = backend_optimizer.Optimizer()

        # Add keyframe to the backend
        self.add_keyframe(agent.start_pose, start_node_id)

        # Best Guess of origin for graph
        new_agent['graph'].add_node(start_node_id, pose=[0,0,0], KF=agent.start_pose)
        new_agent['optimizer'].new_graph(start_node_id, agent.id)

        self.graphs[agent.id] = new_agent

        self.num_agents += 1


    def add_keyframe(self, KF, node_id):
        self.new_keyframes.append([node_id, KF[0], KF[1], KF[2]])

    def add_odometry(self, vehicle_id, from_id, to_id, covariance, transform, keyframe):

        # Figure out which graph this odometry goes with
        keys = self.graphs.keys()
        count = 0
        for id in keys:
            if vehicle_id in self.graphs[id]['connected_agents']:
                count += 1
                graph = self.graphs[id]
                graph['graph'].add_edge(from_id, to_id, covariance=covariance,
                                        transform=transform, from_id=from_id, to_id=to_id)
                graph['graph'].node[to_id]['KF'] = keyframe

                # Add Keyframe to the backend
                self.add_keyframe(keyframe, to_id)

                # Concatenate to get a best guess for the pose of the node
                from_pose = graph['graph'].node[from_id]['pose']
                to_pose = self.concatenate_transform(from_pose, transform)
                graph['graph'].node[to_id]['pose'] = to_pose

                # Prepare this edge for the optimizer
                out_node = [to_id, to_pose[0], to_pose[1], to_pose[2]]
                out_edge = [from_id, to_id, transform[0], transform[1], transform[2],
                            covariance[0][0], covariance[1][1], covariance[2][2]]
                graph['node_buffer'].append(out_node)
                graph['edge_buffer'].append(out_edge)

        # Find Loop Closures
        # if count > 1:
        #     debug = 1
        self.optimization_counter += 1
        if self.optimization_counter > self.num_agents * 10:
            for id in keys:
                self.update_gtsam(id)

        if len(self.new_keyframes) > 1:
            self.find_loop_closures()

    def finish_up(self):
        # Find loop closures on this latest batch
        self.find_loop_closures()
        # update optimizer
        for id in self.graphs.keys():
            self.update_gtsam(id)


    def update_gtsam(self, id):
        graph = self.graphs[id]
        # if not nx.is_connected(graph['graph']):
        #     print "something went wrong - error 1"
        try:
            self.optimization_time_array.append([id, len(self.graphs), len(graph['graph'].nodes()), len(graph['graph'].nodes()), graph['optimizer'].add_edge_batch(graph['node_buffer'], graph['edge_buffer'])])
            graph['node_buffer'] = []
            graph['edge_buffer'] = []
            self.optimization_time_array.append([id, len(self.graphs), len(graph['graph'].nodes()), len(graph['graph'].nodes()), graph['optimizer'].optimize()])

        except:
            print "something went wrong - error 2"
        optimized_values = graph['optimizer'].get_optimized()
        for node in optimized_values["nodes"]:
            node_id = node[0]
            graph['graph'].node[node_id]['pose']=[node[1], node[2], node[3]]
        self.optimization_counter = 0


    def find_loop_closures(self):
        # Create a new KDTree with all our keyframes
        self.keyframe_matcher.add_points(self.new_keyframes)
        self.keyframes.append(self.new_keyframes)

        # Search for loop closures
        for keyframe in self.new_keyframes:
            to_keyframe = self.keyframe_matcher.find_closest_point(keyframe, self.LC_threshold)

            # this is a bug in the keyframe matcher
            if np.linalg.norm(np.array(keyframe[1:]) - np.array(to_keyframe[1:])) > self.LC_threshold:
                continue

            if to_keyframe[0] != u'none':

                to_node_id = to_keyframe[0]
                from_node_id = keyframe[0]

                # Save off vehicle ids
                to_vehicle_id = int(to_node_id.split("_")[0])
                from_vehicle_id = int(from_node_id.split("_")[0])

                # Get loop closure edge transform
                transform = self.find_transform(keyframe[1:], to_keyframe[1:])

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
                    print "error code 3"

                # If these are the same graph, then just add a loop closure
                if to_graph_id == from_graph_id:
                    self.graphs[to_graph_id]['graph'].add_edge(from_node_id, to_node_id, covariance=covariance,
                                                               transform=transform, from_id=from_node_id,
                                                               to_id=to_node_id)
                    # Add the loop closure to the GTSAM optimizer
                    opt_edge = [from_node_id, to_node_id, transform[0], transform[1], transform[2],
                                covariance[0][0], covariance[1][1], covariance[2][2]]
                    self.graphs[to_graph_id]['edge_buffer'].append(opt_edge)

                # otherwise we need to merge these two graphs
                else:
                    # Figure out which graph to keep
                    if to_graph_id < from_graph_id:
                        graph1 = to_graph_id
                        graph2 = from_graph_id
                    else:
                        graph1 = from_graph_id
                        graph2 = to_graph_id

                    # self.plot()

                    # Merge graph2 into graph1
                    self.update_gtsam(graph2)
                    self.update_gtsam(graph1)
                    self.merge_graphs(self.graphs[graph1], self.graphs[graph2], to_node_id, from_node_id, transform,
                                      covariance)

                    # Now that we have merged the graphs, we can get rid of graph2
                    del self.graphs[graph2]
                    print "deleting graph ", graph2

        # Clear the new keyframes list so we don't keep adding redundant keyframes to the matcher
        self.new_keyframes = []


    # this function merges graph1 into graph2
    def merge_graphs(self, graph1, graph2, to_node_id, from_node_id, transform, covariance):
        # Figure out the transform between graph origins
        transform_between_graphs = []
        if from_node_id in graph1['graph'].node:
            transform_to_lc_node = self.concatenate_transform(graph1['graph'].node[from_node_id]['pose'], transform)
            transform_between_graphs = self.concatenate_transform(transform_to_lc_node, self.invert_transform(
                                                                  graph2['graph'].node[to_node_id]['pose']))
        else:
            # loop closure was "backwards"
            transform_to_lc_node = self.concatenate_transform(graph1['graph'].node[to_node_id]['pose'],
                                                              self.invert_transform(transform))
            transform_between_graphs = self.concatenate_transform(transform_to_lc_node,
                                                                  self.invert_transform(
                                                                      graph2['graph'].node[from_node_id]['pose']))

        # Make a list of new nodes
        new_nodes = [n for n in graph2['graph'].nodes() if n not in graph1['graph'].nodes()]

        # Copy nodes to graph1
        graph1['graph'].add_nodes_from(new_nodes)

        merged_graph = nx.compose(graph1['graph'], graph2['graph'])
        # graph1['graph'].add_edges_from(new_edges)

        # Find the new edges by comparing the merged graph to the graph with all the nodes
        new_edge_graph = nx.difference(merged_graph, graph1['graph'])

        # move the merged graph into graph1, now that we have the new edges
        graph1['graph'] = merged_graph

        # Merge the list of connected agents
        graph1['connected_agents'].extend(graph2['connected_agents'])

        # Add the loop closure to the graph and optimizer
        graph1['graph'].add_edge(from_node_id, to_node_id, covariance=covariance, transform=transform,
                                 from_id=from_node_id, to_id=to_node_id)
        graph1['edge_buffer'].append([from_node_id, to_node_id, transform[0], transform[1], transform[2], covariance[0][0],
                                    covariance[1][1], covariance[2][2]])

        # Transform new nodes to the right coordinate frame, and add them to the optimizer
        for node_id in new_nodes:
            pose = self.concatenate_transform(transform_between_graphs, graph2['graph'].node[node_id]['pose'])
            graph1['graph'].node[node_id]['pose'] = pose
            graph1['graph'].node[node_id]['KF'] = graph2['graph'].node[node_id]['KF']
            graph1['node_buffer'].append([node_id, pose[0], pose[1], pose[2]])

        # Add the new edges to the optimizer
        for edge in new_edge_graph.edges():
            edge_map = graph2['graph'].edge[edge[0]][edge[1]]
            graph1['graph'].edge[edge[0]][edge[1]] = edge_map
            try:
                graph1['edge_buffer'].append([edge_map['from_id'], edge_map['to_id'], edge_map['transform'][0],
                                     edge_map['transform'][1], edge_map['transform'][2],
                                     edge_map['covariance'][0][0], edge_map['covariance'][1][1],
                                     edge_map['covariance'][2][2]])
            except:
                print "error 5"


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

    def init_plots(self, num_subplots):
        self.num_subplots = num_subplots

        rows = int(ceil(self.num_subplots/self.num_subplots**0.5))
        cols = int(ceil(self.num_subplots/float(rows)))


        self.figs = []
        self.axes = []
        self.lines = []
        for i in range(2):
            self.figs.append(plt.figure(i+1, figsize=(12, 15), dpi=80, facecolor='w', edgecolor='k'))
            plt.clf()
            subplot_axes = []
            subplot_lines = []
            for j in range(self.num_subplots):
                subplot_axes.append(plt.subplot(rows, cols, j + 1))
                subplot_lines.append(subplot_axes[-1].plot([0,0]))
            self.axes.append(subplot_axes)
            self.lines.append(subplot_lines)
            self.figs[i].show()
            self.figs[i].canvas.draw()


    def plot(self):
        num_subplots = len(self.graphs) + 1
        if num_subplots < self.num_subplots:
            self.init_plots(num_subplots)

        rows = int(ceil(self.num_subplots/self.num_subplots**0.5))
        cols = int(ceil(self.num_subplots/float(rows)))

        # Plot the combined graph
        for i in range(2):
            combined_graph = nx.Graph()
            for id, graph in self.graphs.iteritems():
                combined_graph = nx.compose(combined_graph, graph['graph'])
            self.plot_graph(combined_graph, title='full truth', edge_color='r', truth=1,
                            axis_handle=self.axes[i][0], line_handle=self.lines[i][0][0], figure_handle=self.figs[i])
            self.figs[i].canvas.flush_events()

        names = [' truth', ' optimized']
        truth = [1, 0]

        agents = []
        if len(self.graphs) > self.num_subplots:
            agents = [[id, len(graph['connected_agents'])] for id, graph in self.graphs.iteritems()]
            agents = sorted(agents, key=itemgetter(1), reverse=True)
            agents = [x[0][0] for x in zip(agents)]
        else:
            agents = [id for id, graph in self.graphs.iteritems()]


        for j in range(2):
            for i in range(self.num_subplots - 1):
                agent = agents[i]
                self.plot_graph(self.graphs[agent]['graph'], title=str(agent) + names[j], edge_color='g', truth=truth[j],
                          axis_handle=self.axes[j][i+1], line_handle=self.lines[j][i+1][0], figure_handle=self.figs[j])
            # self.figs[j].canvas.draw()
            self.figs[j].canvas.draw()
            debug = 1


        # Save frames for future movie making
        self.figs[0].savefig("movie/truth_" + str(self.frame_number).zfill(4) + ".png")
        self.figs[1].savefig("movie/optimized_" + str(self.frame_number).zfill(4) + ".png")
        self.frame_number += 1
        plt.ioff()

        # also pickle up the optimization_time array
        pickle.dump(self.optimization_time_array, open('time_array.pkl', 'wb'))


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

    def plot_graph(self, graph, line_handle, figure_handle, axis_handle, title='default', name='default',
                   arrows=False,  edge_color='m', truth=False, plot_lc=False, ):
        # if axis_handle < 0 or figure_handle < 0:
        #     figure_handle = plt.figure(figsize=(12,15), dpi=80, facecolor='w', edgecolor='k')
        #     figure_handle.clf()
        #     axis_handle = figure_handle.subplot(111)
        axis_handle.set_title(title)

        # if line_handle < 0:
        #     line_handle = plt.plot([0,0])

        path_data = []
        if truth:
            path_data = [[graph.node[node]['KF'][1], graph.node[node]['KF'][0], node] for node in sorted(graph.nodes())]
        else:
            path_data = [[graph.node[node]['pose'][1], graph.node[node]['pose'][0], node] for node in sorted(graph.nodes())]

        # Add nans at discontinuities
        i = 0
        while i  < len(path_data) - 2:
            # If there is no edge between these nodes
            if path_data[i+1][2] not in graph.edge[path_data[i][2]].keys():
                # Insert a Nan to tell matplotlib not to plot a line between these two points
                path_data.insert(i+1, [np.nan, np.nan, 'none'])
                i += 1
            i += 1

        # Plot the points
        path_data = np.array(path_data)
        line_handle.set_xdata(path_data[:,0])
        line_handle.set_ydata(path_data[:,1])
        # axis_handle.draw_artist(axis_handle.patch)
        axis_handle.draw_artist(line_handle)
        # figure_handle.canvas.update()
        axis_handle.relim()
        axis_handle.autoscale_view()
        axis_handle.set_aspect('equal','datalim')

        # # Get positions of all nodes
        # plot_positions = dict()
        # for (i, n) in graph.node.iteritems():
        #     if truth:
        #         plot_positions[i] = [n['KF'][1], n['KF'][0]]
        #     else:
        #         try:
        #             plot_positions[i] = [n['pose'][1], n['pose'][0]]
        #         except:
        #             print "error 6"
        #
        # plot_graph = graph.copy()
        #
        # # Remove Loop Closures
        # if not plot_lc:
        #     for pair in graph.edges():
        #         i = pair[0]
        #         j = pair[1]
        #         vID_i = int(i.split("_")[0])
        #         vID_j = int(j.split("_")[0])
        #         if vID_i != vID_j:
        #             plot_graph.remove_edge(i, j)
        #
        # nx.draw_networkx(plot_graph, pos=plot_positions,
        #                  with_labels=False, ax=axis_handle, edge_color=edge_color,
        #                  linewidths="0.3", node_color='c', node_shape='')
        # if arrows:
        #     for i, n in graph.node.iteritems():
        #         if 'pose' in n:
        #             pose = n['pose']
        #         else:
        #             pose = n['KF']
        #         arrow_length = 1.0
        #         dx = arrow_length * cos(pose[2])
        #         dy = arrow_length * sin(pose[2])
        #         # Be sure to convert to NWU for plotting
        #         axis_handle.arrow(pose[1], pose[0], dy, dx, head_width=arrow_length*0.15, head_length=arrow_length*0.3, fc='c', ec='b')

        # plt.axis("equal")

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









