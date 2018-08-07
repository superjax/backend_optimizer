import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from REO import invert_transform, concatenate_transform, invert_edges, get_global_pose, REO_opt
from GPO import GPO
import os, subprocess
import json


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
    # Error is on line 229 of backend_optimizer.cpp
    REO_optimized, REO_iters = REO_opt(edges, nodes, '0_000', 25, 1e-8)
    GPO_optimized, GPO_iters = gpo.opt(edges, nodes, '0_000', 25, 1e-8)  # This throws and error when calling the optimize function
    # REO_optimized, REO_iters = REO_opt(edges, nodes, '0_000', 25, 1e-8)
    results_dict['GPO_opt'].append(GPO_optimized)
    results_dict['REO_opt'].append(REO_optimized)
    results_dict['truth'].append(truth)


    # # Calculate Error  Right now these don't mean much
    # initial_error = np.sum(norm(global_state[i, 0:2, :] - truth[0:2, :], axis=0))
    # REO_error = np.sum(norm(REO_optimized[0:2, :] - truth[0:2, :], axis=0))
    # diff_error = np.sum(norm(REO_optimized[0:2, :] - GPO_optimized[0:2, :], axis=0))
    # GPO_error = np.sum(norm(GPO_optimized[0:2, :] - truth[0:2, :], axis=0))\
    #
    # results_dict['REO_errors'].append(REO_error)
    # results_dict['GPO_errors'].append(GPO_error)
    # results_dict['diff_errors'].append(diff_error)
    # results_dict['REO_iters'].append(REO_iters)
    # results_dict['GPO_iters'].append(GPO_iters)
    #
    # if REO_error < 0.01:
    #     results_dict['num_REO_correct'] += 1
    # if GPO_error < 0.01:
    #     results_dict['num_GPO_correct'] += 1
    #
    # return results_dict
    return REO_optimized, GPO_optimized

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
    reo_f, gpo_f = run(data)

    plt.figure(1)
    plt.plot(reo_f[0, :], reo_f[1, :])  # Currently the REO_optimization doesn't do anything
    # for i, loop in enumerate(lc):
    #     plt.plot(results[1, loop], results[2, loop], 'r')  # plot the loop closures

    plt.figure(2)
    plt.plot(gpo_f[0, :], gpo_f[1, :])  # Currently the GPO_optimization doesn't do anything either
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
