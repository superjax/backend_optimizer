import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from REO import invert_transform, concatenate_transform, invert_edges, get_global_pose, REO_opt
from GPO import GPO
import os, subprocess
import json
import time


def norm(v, axis=None):
    return np.sqrt(np.sum(v*v, axis=axis))


def run(data):
    edges = data['edges']
    nodes = data['nodes']
    global_state = data['global_state']
    truth = np.array(data['truth'])

    # For each agent, optimize with REO and GTSAM
    results_dict = dict()

    results_dict['REO_errors'] = []
    results_dict['GPO_errors'] = []
    results_dict['diff_errors'] = []
    results_dict['REO_iters'] = []
    results_dict['GPO_iters'] = []
    results_dict['num_REO_correct'] = 0
    results_dict['num_GPO_correct'] = 0
    results_dict['edges'] = []
    results_dict['nodes'] = []
    results_dict['REO_opt'] = []
    results_dict['GPO_opt'] = []
    results_dict['truth'] = []

    gpo = GPO()

    # Optimize with both optimizers
    results_dict['edges'].append(edges)
    results_dict['nodes'].append(nodes)

    t0_g = time.time()
    GPO_optimized, GPO_iters = gpo.opt(edges, nodes, '0_000', 25, 1e-8)  # This throws and error when calling the optimize function
    tf_g = time.time()
    dt_g = tf_g - t0_g

    t0_r = time.time()
    REO_optimized, REO_iters = REO_opt(edges, nodes, '0_000', 25, 1e-8)
    tf_r = time.time()
    dt_r = tf_r - t0_r

    results_dict['GPO_opt'] = GPO_optimized
    results_dict['REO_opt'] = REO_optimized
    results_dict['GPO_Time'] = dt_g
    results_dict['REO_Time'] = dt_r

    return results_dict

if __name__ == '__main__':
    subprocess.Popen(['mkdir', '-p', 'tests/well_conditioned/plots_hw'])

    f = open('data1.txt', 'r')
    data = json.load(f)
    edges = data['edges']
    nodes = data['nodes']
    lc = data['loop_closures']
    for i in range(len(nodes)):
        nodes[i][0] = str(nodes[i][0]).zfill(3)

    cwd = os.getcwd()
    os.chdir("tests/well_conditioned")

    print( "running optimization")
    results = run(data)
    reo_f = results['REO_opt']
    gpo_f = results['GPO_opt']
    print('GPO Time: ', results['GPO_Time'])
    print('REO Time: ', results['REO_Time'])

    plt.figure(1)
    plt.plot(reo_f[0, :], reo_f[1, :], label='REO', color='b')
    for i, loop in enumerate(lc):
        plt.plot(reo_f[0, loop], reo_f[1, loop], 'r')  # plot the loop closures.
    plt.axis([-20, 20, -3, 38])
    plt.legend(['REO Path', 'Loop closures'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False,
               shadow=False, ncol=2)

    plt.savefig("plots_hw/reo_hw.svg", bbox_inches='tight', pad_inches=0)

    plt.figure(2)
    plt.plot(gpo_f[0, :], gpo_f[1, :], label='GPO', color='b')
    for i, loop in enumerate(lc):
        plt.plot(gpo_f[0, loop], gpo_f[1, loop], 'r')  # plot the loop closures.
    plt.axis([-20, 20, -3, 38])
    plt.legend(['GPO Path', 'Loop closures'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False, shadow=False, ncol=2)
    plt.savefig("plots_hw/gpo_hw.svg", bbox_inches='tight', pad_inches=0)
    plt.show()


    # reo_f['avg_REO_error'] = sum(reo_f['REO_errors']) / float(reo_f['num_robots'])
    # reo_f['avg_GPO_error'] = sum(reo_f['GPO_errors']) / float(reo_f['num_robots'])
    # reo_f['avg_REO_iter'] = float(sum(reo_f['REO_iters'])) / float(reo_f['num_robots'])
    # reo_f['avg_GPO_iter'] = float(sum(reo_f['GPO_iters'])) / float(reo_f['num_robots'])
    # reo_f['max_REO_error'] = max(reo_f['REO_errors'])
    # reo_f['max_GPO_error'] = max(reo_f['GPO_errors'])
    #
    #
    # # Plot all the trajectories
    # print( "plotting trajectories")
    # plt.figure(2, figsize=(12,9))
    # plt.set_cmap('Set1')
    # for j, (REO, GPO, nodes, truth) in tqdm(enumerate(zip(reo_f['REO_opt'], reo_f['GPO_opt'], reo_f['nodes'], reo_f['truth'])), total=reo_f['num_robots']):
    #     initial_pos = np.array([[nodes[i][1], nodes[i][2]] for i in range(len(nodes))])
    #     plt.clf()
    #     ax=plt.subplot(111)
    #
    #     plt.plot(initial_pos[:,0], initial_pos[:,1], label='initial', linewidth=3, dashes=[10, 5], alpha = 0.5, color='g')
    #     plt.plot(truth[0,:], truth[1,:], label="truth", linewidth=1, alpha=1, color='k')
    #     plt.plot(REO[0,:], REO[1,:], label='REO', linewidth=3,  alpha=0.8, dashes=[4,2], color='b')
    #     plt.plot(GPO[0, :], GPO[1, :], label='GPO', linewidth=3, alpha=0.8, dashes=[2, 4], color='r')
    #     box = ax.get_position()
    #     ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False, shadow=False, ncol=4)
    #     plt.savefig("plots_hw/traj" + str(j).zfill(3) + ".svg", bbox_inches='tight', pad_inches=0)
    #     if j > 100:
    #         break
