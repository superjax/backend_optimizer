#!/usr/bin/env python

import rosbag
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import math


def get_edges(msgs, begin, end, l, flipped):
    edges = []
    lc = []
    for i in range(begin, end):
        to_id = msgs[i].to_node_id
        from_id = msgs[i].from_node_id
        x = msgs[i].transform.translation.x
        y = msgs[i].transform.translation.y
        n1 = msgs[i].transform.rotation.x
        n2 = msgs[i].transform.rotation.y
        n3 = msgs[i].transform.rotation.z
        n4 = msgs[i].transform.rotation.w
        theta = math.atan2(2 * (n4 * n3 + n1 * n2), 1 - 2 * (n2 ** 2 + n3 ** 2))  # conversion from wikipedia
        if to_id - from_id != 1:
            lc.append([msgs[i].from_node_id, msgs[i].to_node_id])
            cov1 = 1e5
            cov2 = 1e5
            cov3 = 1e3
        else:
            cov1 = 1.0 / msgs[i].covariance[0]
            cov2 = 1.0 / msgs[i].covariance[7]
            cov3 = 1.0 / msgs[i].covariance[35]
        edges.append(['0_' + str(from_id).zfill(3), '0_' + str(to_id).zfill(3), x, y, theta, cov1, cov2, cov3])

        # plot the path
        x = [['0_' + str(l).zfill(3), 0, 0, 0]]
        x2 = [[0, 0, 0, 0]]
        if flipped:  # flip the initial orientation of the second robot
            x2[0][3] = np.pi
            x[0][3] = np.pi
        i = 0
        for edge in edges:
            if int(edge[1].split("_")[1]) - int(edge[0].split("_")[1]) == 1:
                x1 = x[i][1] + edge[2] * np.cos(x[i][3]) - edge[3] * np.sin(x[i][3])
                y1 = x[i][2] + edge[2] * np.sin(x[i][3]) + edge[3] * np.cos(x[i][3])
                phi1 = x[i][3] + edge[4]
                if phi1 > np.pi:
                    phi1 -= 2. * np.pi
                elif phi1 < -np.pi:
                    phi1 += 2. * np.pi
                x.append(['0_' + str(i + l + 1).zfill(3), x1, y1, phi1])  # need to adjust this
                x2.append([i + 1, x1, y1, phi1])
                i += 1

    return edges, x, x2, lc


if __name__ == "__main__":
    inbag = rosbag.Bag('/home/brendon/Documents/backend_optimizer/python/bags/fletcher.bag')
    # THIS PARSES THE double_loop_fletcher.bag (something like that) in the directory in the j drive group folder
    # groups/magicc-data/old_server/groups/relative_nav/rosbag/FLETCHER

    msgs = []
    for topic, msg, t in tqdm(inbag.read_messages(), total=inbag.get_message_count()):
        if topic == "/edge" or topic == "/loop_closure":
            msgs.append(msg)

    l = int(math.floor(len(msgs) / 2.0))  # this is the node that splits what is covered by each robot

    plot = True
    edges1, nodes1, x,  lc1 = get_edges(msgs, 0, l, 0, False)  # get edges for the first robot
    edges2, nodes2, x2, lc2 = get_edges(msgs, l, len(msgs), len(nodes1), True)

    # Is this what I want to do?
    edges = edges1
    for edge in edges2:
        edges.append(edge)
    nodes = nodes1
    for node in nodes2:
        nodes1.append(node)

    data = dict()
    data['edges'] = edges
    data['nodes'] = nodes
    data['edges1'] = edges1
    data['edges2'] = edges2
    data['nodes1'] = nodes1
    data['nodes2'] = nodes2
    data['lc1'] = lc1
    data['lc2'] = lc2
    data['truth'] = nodes
    data['global'] = nodes

    filename = "data2.txt"
    f = open(filename, 'w')
    json.dump(data, f)
    f.close()

    if plot:
        t = len(x)
        plt.figure(1)
        x = np.array(x)
        x2 = np.array(x2)
        plt.plot(x[:, 1], x[:, 2], color='b')
        plt.plot(x2[:, 1], x2[:, 2], color='k')

        for i, loop in enumerate(lc1):
            plt.plot(x[loop, 1], x[loop, 2], 'r')  # plot the loop closures


        for i, loop in enumerate(lc2):
            if loop[0] < t:
                loop[1] -= l
                plt.plot([x[loop[0], 1], x2[loop[1], 1]], [x[loop[0], 2], x2[loop[1], 2]], color='r')  # plot the loop closures
            elif loop[1] < t:
                loop[0] -= l
                plt.plot([x2[loop[0], 1], x[loop[1], 1]], [x2[loop[0], 2], x[loop[1], 2]], color='r')  # plot the loop closures
            else:
                plt.plot(x2[loop, 1], x2[loop, 2], 'r')  # plot the loop closures

        # plt.axis([-20, 20, -3, 38])
        plt.legend(['Robot 1', 'Robot 2', 'Loop closures'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False,
                   shadow=False, ncol=3)

        plt.savefig("tests/well_conditioned/plots_hw/data_multi.eps", bbox_inches='tight', format='eps', pad_inches=0)
        plt.show()


