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
    results_dict['REO_iters'] = REO_iters
    results_dict['GPO_iters'] = GPO_iters

    return results_dict


if __name__ == '__main__':
    subprocess.Popen(['mkdir', '-p', 'tests/well_conditioned/plots_hw'])

    f = open('data3.txt', 'r')
    data = json.load(f)
    num_nodes= data['num_nodes']

    cwd = os.getcwd()
    os.chdir("tests/well_conditioned")

    print( "running optimization")
    results = run(data)
    reo_f = results['REO_opt']
    gpo_f = results['GPO_opt']
    print('GPO Time: ', results['GPO_Time'])
    print('REO Time: ', results['REO_Time'])
    print('GPO Iters: ', results['GPO_iters'])
    print('REO Iters: ', results['REO_iters'])

    colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y', 'r--', 'b--', 'g--']

    plt.figure(1)
    for i in range(len(num_nodes)):
        if i == 0:
            plt.plot(reo_f[0, 0:num_nodes[i]], reo_f[1, 0:num_nodes[i]], label='REO', color=colors[i % len(colors)])
        else:
            plt.plot(reo_f[0, i * num_nodes[i - 1]:(i + 1) * num_nodes[i]],
                     reo_f[1, i * num_nodes[i - 1]:(i + 1) * num_nodes[i]], label='GPO',
                     color=colors[i % len(colors)])
    # for i, loop in enumerate(lc):
    #     plt.plot(reo_f[0, loop], reo_f[1, loop], 'r')  # plot the loop closures.
    # plt.axis([-20, 20, -3, 38])
    # plt.legend(['Robot 1', 'Robot 2', 'Robot 3'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False,
    #            shadow=False, ncol=3)
    plt.savefig("demo_2_plots/reo_data.eps", bbox_inches='tight', format='eps', pad_inches=0)


    plt.figure(2)
    for i in range(len(num_nodes)):
        if i == 0:
            plt.plot(gpo_f[0, 0:num_nodes[i]], gpo_f[1, 0:num_nodes[i]], label='REO', color=colors[i % len(colors)])
        else:
            plt.plot(gpo_f[0, i*num_nodes[i - 1]:(i+1)*num_nodes[i]], gpo_f[1, i*num_nodes[i - 1]:(i+1)*num_nodes[i]], label='GPO',
                     color=colors[i % len(colors)])
    # plt.legend(['Robot 1', 'Robot 2', 'Robot 3'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False,
    #            shadow=False, ncol=3)
    plt.savefig("demo_2_plots/pgo_data.eps", bbox_inches='tight', format='eps', pad_inches=0)
    plt.show()