from keyframe_matcher_sim import KeyframeMatcher
from robot import *
from controller import *
from tqdm import tqdm
import os
import time
import json


if __name__ == "__main__":
    stamp = time.time()
    # if not os.path.isdir("tests/well_conditioned/demo_2_plots"):
    #     os.mkdir("tests/well_conditioned/demo_2_plots")
    # os.chdir("tests/well_conditioned")
    dt = 0.1    
    time = np.arange(0, 100.01, dt) # origninal time was 300
    colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y', 'r--', 'b--', 'g--']

    robots = []
    controllers = []
    num_robots = 3  # Will up to 10 later. Start with 2 while familiarizing myself.
    KF_frequency_s = 1.0
    plot_frequency_s = 10

    start_pose_range = [1, 1, 2] # initial values were 5, 3, 2

    start_poses = [[randint(-start_pose_range[0], start_pose_range[0])*10,
                    (-1)**r * start_pose_range[1]*10,
                    -np.pi/2.0 if r % 2 == 0 else np.pi/2.0] for r in range(num_robots)]

    P_perfect = np.array([[0.00001, 0, 0], [0, 0.00001, 0], [0, 0, 0.00001]])
    G = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])  # May need to play with covariances a little. Who knows
    lc_cov = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.3]])

    print("simulating robots")

    kf_matcher = KeyframeMatcher(lc_cov)
    controllers = [Controller(start_poses[r]) for r in range(num_robots)]
    robots = [Robot(r, G, start_poses[r]) for r in range(num_robots)]

    lc = []

    for t in tqdm(time):
        for r in range(num_robots):
            # Run each robot through the trajectory
            u = controllers[r].control(t, robots[r].state(), [start_pose_range[0]*10+5, start_pose_range[1]*10+5])
            robots[r].propagate_dynamics(u, dt)
            if t % KF_frequency_s == 0 and t > 0:
                # Declare a new keyframe
                edge, KF = robots[r].reset()

                # Add the keyframe to the kf_matcher
                kf_matcher.add_keyframe(KF, str(r) + "_" + str(robots[r].keyframe_id()).zfill(3))

        # plot maps
        if t % plot_frequency_s == 0 and t > 0:
            # look for loop closures
            loop_closures = kf_matcher.find_loop_closures()
            for loop in loop_closures:
                lc.append(loop)


    # Formatting Data for exporting to a file
    edges = []
    nodes = []
    truth = []
    num_nodes = []
    for r in range(num_robots):
        num_nodes.append(len(robots[r].nodes))
        for e in robots[r].edges:
            edges.append(e)
        for n in robots[r].nodes:
            nodes.append(n)
        for n in robots[r].I_nodes:
            truth.append(n)
        for l in lc:
            from_id = l['from_node_id']
            to_id = l['to_node_id']
            tf = l['transform']
            cov = l['covariance']
            edges.append([from_id, to_id, tf[0], tf[1], tf[2], 1.0/cov[0, 0], 1.0/cov[1, 1], 1.0/cov[2, 2]])

    print(len(edges))
    print(len(nodes))


    data = dict()
    data['edges'] = edges
    data['nodes'] = nodes
    # data['lcs'] = lc find a way to dump this
    data['truth'] = truth
    data['global'] = truth
    data['num_nodes'] = num_nodes

    filename = 'data3.txt'
    f = open(filename, 'w')
    json.dump(data, f)
    f.close()

    # Plot the noisy initial data
    plt.figure(1)
    for r in range(num_robots):
        plt.plot(np.array(robots[r].plt_nodes)[:, 1], np.array(robots[r].plt_nodes)[:, 2], colors[r % len(colors)])

    plt.legend(['Robot 1', 'Robot 2', 'Robot 3'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False,
               shadow=False, ncol=3)
    plt.savefig("tests/well_conditioned/demo_2_plots/noisy_data.eps", bbox_inches='tight', format='eps', pad_inches=0)

    # Plot the true initial data
    plt.figure(2)
    for r in range(num_robots):
        plt.plot(np.array(robots[r].true_nodes)[:, 1], np.array(robots[r].true_nodes)[:, 2], colors[r % len(colors)])
    plt.legend(['Robot 1', 'Robot 2', 'Robot 3'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False,
               shadow=False, ncol=3)
    plt.savefig("tests/well_conditioned/demo_2_plots/true_data.eps", bbox_inches='tight', format='eps', pad_inches=0)

    # Plot the true map
    plt.figure(3)
    for r in range(num_robots):
        plt.plot(np.array(robots[r].I_nodes)[:, 1], np.array(robots[r].I_nodes)[:, 2], colors[r%len(colors)])
    plt.legend(['Robot 1', 'Robot 2', 'Robot 3'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False,
               shadow=False, ncol=3)
    plt.savefig("tests/well_conditioned/demo_2_plots/inertial_data.eps", bbox_inches='tight', format='eps', pad_inches=0)
    plt.show()

    debug = 1