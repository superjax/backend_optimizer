#!/usr/bin/env python

import rosbag
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import math

def get_edges(plot):
    # inbag = rosbag.Bag('/home/brendon/Documents/backend_optimizer/python/bags/WSC_02.bag')
    inbag = rosbag.Bag('/home/brendon/Documents/backend_optimizer/python/bags/fletcher.bag')
    # THIS PARSES THE double_loop_fletcher.bag (something like that) in the directory in the j drive group folder
    # groups/magicc-data/old_server/groups/relative_nav/rosbag/FLETCHER

    edges = []
    lc = []

    for topic, msg, t in tqdm(inbag.read_messages(), total=inbag.get_message_count()):
        # if topic not in topics:
        #     topics.append(topic)

        if topic == "/edge":  # Will I have to calculate the odometry to put into the list?
            to_id = msg.to_node_id
            from_id = msg.from_node_id
            x = msg.transform.translation.x
            y = msg.transform.translation.y
            n1 = msg.transform.rotation.x
            n2 = msg.transform.rotation.y
            n3 = msg.transform.rotation.z
            n4 = msg.transform.rotation.w
            theta = math.atan2(2 * (n4 * n3 + n1 * n2), 1 - 2 * (n2 ** 2 + n3 ** 2))  # conversion from wikipedia
            cov1 = 1.0/msg.covariance[0]
            cov2 = 1.0/msg.covariance[7]
            cov3 = 1.0/msg.covariance[35]
            edges.append(['0_' + str(from_id).zfill(3), '0_'+str(to_id).zfill(3), x, y, theta, cov1, cov2, cov3])  # Does this need to be velocity

        if topic == "/loop_closure":  # add loop closures to the edge list
            to_id = msg.to_node_id
            from_id = msg.from_node_id
            x = msg.transform.translation.x
            y = msg.transform.translation.y
            n1 = msg.transform.rotation.x
            n2 = msg.transform.rotation.y
            n3 = msg.transform.rotation.z
            n4 = msg.transform.rotation.w
            theta = math.atan2(2 * (n4 * n3 + n1 * n2), 1 - 2 * (n2 ** 2 + n3 ** 2))
            cov1 = 1e5
            cov2 = 1e5
            cov3 = 1e3
            lc.append([[msg.from_node_id], [msg.to_node_id]])
            edges.append(['0_' + str(from_id).zfill(3), '0_' + str(to_id).zfill(3), x, y, theta, cov1, cov2, cov3])

    # plot the path
    x = [['0_000', 0, 0, 0]]
    x2 =[[0, 0, 0, 0]]
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
            x.append(['0_'+str(i+1).zfill(3), x1, y1, phi1])
            x2.append([i+1, x1, y1, phi1])
            i += 1

    if plot:
        plt.figure(1)
        x2 = np.array(x2)
        plt.plot(x2[:, 1], x2[:, 2])
        for i, loop in enumerate(lc):
            plt.plot(x2[loop, 1], x2[loop, 2], 'r')  # plot the loop closures
        plt.axis([-20, 20, -3, 38])
        plt.legend(['Initial Data', 'Loop closures'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False,
                   shadow=False, ncol=2, prop={'size' : 14})

        plt.savefig("tests/well_conditioned/plots_hw/data_hw.eps", bbox_inches='tight', format='eps', pad_inches=0)
        plt.show()

    return x, edges, lc


if __name__ == "__main__":
    plot = True
    poses, edges, lcs = get_edges(plot)

    truth = poses  # is this the right thing to do with the lack of perfect information
    global_state = poses  # assuming the local and global are the same frame: Not entirely needed for this

    data = dict()
    data['edges'] = edges
    data['nodes'] = poses
    data['truth'] = truth
    data['global_state'] = global_state
    data['loop_closures'] = lcs
    filename = "data1.txt"
    f = open(filename, 'w')
    json.dump(data, f)
    f.close()
