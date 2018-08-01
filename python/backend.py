#!/usr/bin/env python
# -*- coding: utf-8 -*-
import backend_optimizer
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from math import *
import numpy as np
import pickle
from threading import Lock
import os


# class Node():
#     def __init__(self, id, pose):
#         self.id = id
#         self.pose = pose
#         self.true_pose = pose
#
#     def set_pose(self, pose):
#         self.pose = pose

class Backend():
    def __init__(self):
        self.graphs = dict()

        self.edge_buffer_for_optimizer = []
        self.lc_buffer_for_optimizer = []
        self.node_buffer_for_optimizer = []

        self.frame_number = 0

        self.special_agent_id = 0

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

        self.max_subplots = 5

        self.graphs_mutex = Lock()

        if not os.path.exists("movie"):
            os.mkdir("movie")
        os.chdir("movie")
        os.system('rm *.png')
        os.system('rm *.avi')
        os.chdir("..")


    def add_agent(self, agent):
        # Initialize and keep track of this agent
        start_node_id = str(agent.id) + "_000"
        
        new_agent = dict()
        new_agent['graph'] = nx.Graph()
        new_agent['connected_agents'] = [agent.id]
        new_agent['node_buffer'] = []
        new_agent['edge_buffer'] = []
        new_agent['optimizer'] = backend_optimizer.Optimizer()

        # Best Guess of origin for graph
        new_agent['graph'].add_node(start_node_id, pose=[0,0,0], KF=agent.start_pose)
        new_agent['optimizer'].new_graph(start_node_id, agent.id)

        self.graphs[agent.id] = new_agent

        self.num_agents += 1

    def add_truth(self, node_id, KF):
        vehicle_id = int(node_id.split("_")[0])
        for id in self.graphs.keys():
            if vehicle_id in self.graphs[id]['connected_agents']:
                self.graphs[id]['graph'].node[node_id]['KF'] = KF

    def add_odometry(self, vehicle_id, from_id, to_id, covariance, transform, keyframe):

        # Figure out which graph this odometry goes with
        keys = self.graphs.keys()
        count = 0
        for id in keys:
            if vehicle_id in self.graphs[id]['connected_agents']:
                count += 1
                graph = self.graphs[id]
                self.graphs_mutex.acquire()
                graph['graph'].add_edge(from_id, to_id, covariance=covariance,
                                        transform=transform, from_id=from_id, to_id=to_id)

                if keyframe:
                    graph['graph'].node[to_id]['KF'] = keyframe

                # Concatenate to get a best guess for the pose of the node
                from_pose = graph['graph'].node[from_id]['pose']
                to_pose = self.concatenate_transform(from_pose, transform)
                graph['graph'].node[to_id]['pose'] = to_pose
                self.graphs_mutex.release()

                # Prepare this edge for the optimizer
                out_node = [to_id, to_pose[0], to_pose[1], to_pose[2]]
                out_edge = [from_id, to_id, transform[0], transform[1], transform[2],
                            covariance[0][0], covariance[1][1], covariance[2][2]]
                graph['node_buffer'].append(out_node)
                graph['edge_buffer'].append(out_edge)

        # Optimize
        # self.optimization_counter += 1
        # if self.optimization_counter > self.num_agents * 10:

    def finish_up(self):
        # update optimizer
        for id in self.graphs.keys():
            self.update_gtsam(id)


    def update_gtsam(self, id):
        if id not in self.graphs:
            return

        graph = self.graphs[id]
        try:
            # Add edges and nodes and run optimization
            self.optimization_time_array.append([id, len(self.graphs), len(graph['graph'].nodes()),
                                                 len(graph['node_buffer']), len(graph['edge_buffer']),
                                                 graph['optimizer'].add_edge_batch(graph['node_buffer'],
                                                                                   graph['edge_buffer'])])
            graph['node_buffer'] = []
            graph['edge_buffer'] = []
            # Run a second optimization step
            # self.optimization_time_array.append([id, len(self.graphs), len(graph['graph'].nodes()), graph['optimizer'].optimize()])

        except:
            print("something went wrong - error 2")
        optimized_values = graph['optimizer'].get_optimized()
        for node in optimized_values["nodes"]:
            node_id = node[0]
            graph['graph'].node[node_id]['pose']=[node[1], node[2], node[3]]
        self.optimization_counter = 0


    def add_loop_closure(self, from_node_id, to_node_id, transform, covariance):
        # Save off vehicle ids
        to_vehicle_id = int(to_node_id.split("_")[0])
        from_vehicle_id = int(from_node_id.split("_")[0])

        # Figure out which graphs these agents come from
        from_graph_id = -1
        to_graph_id = -1
        for id, graph in self.graphs.items():
            if from_vehicle_id in graph['connected_agents']:
                from_graph_id = id
            if to_vehicle_id in graph['connected_agents']:
                to_graph_id = id

        if from_graph_id < 0 or to_graph_id < 0:
            print("error code 3")

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
            self.graphs_mutex.acquire()
            del self.graphs[graph2]
            self.graphs_mutex.release()
            print("deleting graph %d" % graph2)


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
            if 'KF' in graph1['graph'].node[node_id]:
                graph1['graph'].node[node_id]['KF'] = graph2['graph'].node[node_id]['KF']
            graph1['node_buffer'].append([node_id, pose[0], pose[1], pose[2]])

        # Add the new edges to the optimizer
        for edge in new_edge_graph.edges():
            edge_map = graph2['graph'].edges[edge[0],edge[1]]
            graph1['graph'].add_edge(edge[0],edge[1], **edge_map)
            try:
                graph1['edge_buffer'].append([edge_map['from_id'], edge_map['to_id'], edge_map['transform'][0],
                                     edge_map['transform'][1], edge_map['transform'][2],
                                     edge_map['covariance'][0][0], edge_map['covariance'][1][1],
                                     edge_map['covariance'][2][2]])
            except:
                print("error 5")


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

    def plot(self):
        for id in self.graphs.keys():
            self.update_gtsam(id)

        num_subplots = len(self.graphs) + 1

        if num_subplots > self.max_subplots:
            num_subplots = self.max_subplots

        rows = int(ceil(num_subplots/num_subplots**0.5))
        cols = int(ceil(num_subplots/float(rows)))

        # Create combined graph for plotting
        combined_graph = nx.Graph()
        self.graphs_mutex.acquire()
        for id, graph in self.graphs.items():
            combined_graph = nx.compose(combined_graph, graph['graph'])
        self.graphs_mutex.release()

        # Allow rendering
        plt.ion()

        for i in range(2):
            plt.figure(i+1, figsize=(12, 15), dpi=80, facecolor='w', edgecolor='k')
            plt.clf()
            # Full Truth Plots are the first subplot
            ax = plt.subplot(rows, cols, 1)
            self.plot_graph(combined_graph, axis_handle=ax, title='full_truth', truth=True)

            # Now plot individual maps
            self.graphs_mutex.acquire()
            for l, j in enumerate(self.graphs.keys()):
                if l >= self.max_subplots:
                    break
                ax = plt.subplot(rows, cols, l + 2)
                if i == 0:
                    self.plot_graph(self.graphs[j]['graph'], axis_handle=ax, title=str(j) + ' truth', truth=True)
                else:
                    self.plot_graph(self.graphs[j]['graph'], axis_handle=ax, title=str(j) + ' optimized', truth=False)
            self.graphs_mutex.release()

        # Render Image
        plt.show()
        plt.pause(0.001)

        # Save figure
        plt.figure(1).savefig("truth_" + str(self.frame_number).zfill(4) + ".png")
        plt.figure(2).savefig("optimized_" + str(self.frame_number).zfill(4) + ".png")
        self.frame_number += 1

        plt.ioff()

        # also pickle up the optimization_time array
        pickle.dump(self.optimization_time_array, open('time_array.pkl', 'wb'))


    def plot_graph(self, graph, axis_handle, title='default', truth=False):
        axis_handle.set_title(title)


        # create the list of positions of each node, divided by agent
        path_data = dict()
        for agent in range(self.num_agents):
            if truth:
                path_data[agent] = [[graph.node[node]['KF'][1], graph.node[node]['KF'][0], node] for node in sorted(graph.nodes()) if 'KF' in graph.node[node] and agent == int(node.split("_")[0])]
            else:
                path_data[agent] = [[graph.node[node]['pose'][1], graph.node[node]['pose'][0], node] for node in sorted(graph.nodes()) if agent == int(node.split("_")[0])]

            # Add nans at discontinuities
            i = 0
            while i  < len(path_data[agent]) - 2:
                # If there is no edge between these nodes
                if path_data[agent][i][2] not in graph.edges() or path_data[agent][i+1][2] not in graph.edge[path_data[agent][i][2]].keys():
                    # Insert a Nan to tell matplotlib not to plot a line between these two points
                    path_data[agent].insert(i+1, [np.nan, np.nan, 'none'])
                    i += 1
                i += 1

            plot_node_names = False
            if plot_node_names:
                for point in path_data[agent]:
                    if point[2] != 'none':
                        axis_handle.text(point[0], point[1], point[2], fontsize=7)

            # Plot the points
            if len(path_data[agent]) > 2:
                path_data[agent] = np.array(path_data[agent])
                plt.plot(path_data[agent][:,0], path_data[agent][:,1],
                         color = plt.cm.get_cmap('Dark2')(agent/float(self.num_agents)))
                plt.scatter(float(path_data[agent][-1, 0]), float(path_data[agent][-1, 1]),
                         color=plt.cm.get_cmap('Dark2')(agent / float(self.num_agents)), marker='o')

        plt.axis('equal')

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









