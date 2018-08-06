from REO import REO, invert_transform, concatenate_transform
from keyframe_matcher_sim import KeyframeMatcher
from MC_robot import *
from controller import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from REO import invert_transform, concatenate_transform, invert_edges, get_global_pose, REO_opt
from GPO import GPO
import os, subprocess
# from combined import combined_opt

def norm(v, axis=None):
    return np.sqrt(np.sum(v*v, axis=axis))


def generate_house(filename, angle_offset = 0, num_robots = 1000):
    print( "generating", num_robots, "house trajectories and saving to", filename)
    perfect_edges = np.array([[1., 1., 1., 1., 2 ** 0.5, 2 ** 0.5 / 2.0, 2 ** 0.5 / 2.0, 2 ** 0.5],
                              [0., 0., 0., 0., 0, 0, 0, 0],
                              [np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, 3.0 * np.pi / 4.0, np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, 0]])

    x0 = np.array([0, 0, 0])

    dirs = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    Omegas = [np.diag([1e5, 1e5, 1e3]) for i in range(perfect_edges.shape[1])]

    odometry = np.zeros((num_robots, 3, perfect_edges.shape[1]))
    global_estimate = np.zeros((num_robots, 3, perfect_edges.shape[1]+1))
    truth = get_global_pose(perfect_edges, x0.copy())
    # invert_edges(perfect_edges, dirs, [5, 7, 2])

    cycles = [[0, 4],
              [2, 5],
              [1, 7],
              [3, 8],
              [0, 6]]

    lc = np.zeros([3, len(cycles)])
    for i, cycle in enumerate(cycles):
        for j in range(cycle[0], cycle[1]):
            lc[:,i] = concatenate_transform(lc[:,i], perfect_edges[:,j])

    lc_omega = [np.diag([1e5, 1e5, 1e3]) for i in range(lc.shape[1])]

    # Turn off some loop closures
    active_lc = [0, 1, 2, 3, 4]
    lc = lc[:, active_lc]
    lc_omega = [lc_omega[i] for i in active_lc]
    cycles = [cycles[i] for i in active_lc]

    edge_g2o_lists = []
    nodes_g2o_lists = []

    for robot in tqdm(range(num_robots)):
        edge_noise = np.array([[np.random.normal(0, (1. / Omegas[i][0][0])**0.5) for i in range(perfect_edges.shape[1])],
                               [np.random.normal(0, (1. / Omegas[i][1][1])**0.5) for i in range(perfect_edges.shape[1])],
                               [np.random.normal(0, (1. / Omegas[i][2][2])**0.5) for i in range(perfect_edges.shape[1])]])
        edge_noise[:,0] = np.zeros(3) # No noise on first edge (virtual zero edge)
        odometry[robot,:,:] = perfect_edges  + edge_noise
        global_estimate[robot,:,:] = get_global_pose(odometry[robot,:,:], x0.copy())

        # Pack into g2o-style list
        edges = []
        nodes = []
        for i in range(len(perfect_edges.T)):
            if dirs[i] > 0:
                edges.append(['0_'+str(i).zfill(3), '0_'+str(i + 1).zfill(3),
                              odometry[robot,0,i], odometry[robot,1,i], odometry[robot,2,i],
                              Omegas[i][0, 0], Omegas[i][1, 1], Omegas[i][2, 2]])
            else:
                edges.append(['0_' + str(i+1).zfill(3), '0_' + str(i).zfill(3),
                              odometry[robot, 0, i], odometry[robot, 1, i], odometry[robot, 2, i],
                              Omegas[i][0, 0], Omegas[i][1, 1], Omegas[i][2, 2]])

        for i, L in enumerate(lc.T):
                from_num = cycles[i][0]
                to_num = cycles[i][1]
                edges.append(['0_'+str(from_num).zfill(3), '0_'+str(to_num).zfill(3),
                              L[0], L[1], L[2],
                              lc_omega[i][0, 0], lc_omega[i][1, 1], lc_omega[i][2, 2]])
        for i, state in enumerate(global_estimate[robot,:,:].T):
            nodes.append(['0_'+str(i).zfill(3), state[0], state[1], state[2]])

        edge_g2o_lists.append(edges)
        nodes_g2o_lists.append(nodes)

    # edges = perfect_edges
    # edges = noisy_edges

    f = open(filename, 'wb')
    data = dict()
    data['edges'] = edge_g2o_lists
    data['nodes'] = nodes_g2o_lists
    data['global_state'] = global_estimate
    data['truth'] = truth
    pickle.dump(data, f)
    f.close()

def run(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    edges = data['edges']
    nodes = data['nodes']
    global_state = data['global_state']
    truth = np.array(data['truth'])

    num_robots = len(edges)

    # For each agent, optimize with REO and GTSAM
    results_dict = dict()


    results_dict['REO_errors'] = []
    results_dict['GPO_errors'] = []
    results_dict['diff_errors'] = []
    results_dict['REO_iters'] = []
    results_dict['GPO_iters'] = []
    results_dict['num_REO_correct'] = 0
    results_dict['num_GPO_correct'] = 0
    results_dict['num_robots'] = num_robots
    results_dict['edges'] = []
    results_dict['nodes'] = []
    results_dict['REO_opt'] = []
    results_dict['GPO_opt'] = []
    results_dict['truth'] = []

    gpo = GPO()

    for i in tqdm(range(num_robots)):
        # Optimize with both optimizers
        results_dict['edges'].append(edges[i])
        results_dict['nodes'].append(nodes[i])
        GPO_optimized, GPO_iters = gpo.opt(edges[i], nodes[i], '0_000', 25, 1e-8)
        REO_optimized, REO_iters = REO_opt(edges[i], nodes[i], '0_000', 25, 1e-8)
        results_dict['GPO_opt'].append(GPO_optimized)
        results_dict['REO_opt'].append(REO_optimized)
        results_dict['truth'].append(truth)


        # Calculate Error
        initial_error = np.sum(norm(global_state[i, 0:2, :] - truth[0:2, :], axis=0))
        REO_error = np.sum(norm(REO_optimized[0:2, :] - truth[0:2, :], axis=0))
        diff_error = np.sum(norm(REO_optimized[0:2, :] - GPO_optimized[0:2, :], axis=0))
        GPO_error = np.sum(norm(GPO_optimized[0:2, :] - truth[0:2, :], axis=0))\

        # print( "REO:", REO_error, "GPO:", GPO_error)

        results_dict['REO_errors'].append(REO_error)
        results_dict['GPO_errors'].append(GPO_error)
        results_dict['diff_errors'].append(diff_error)
        results_dict['REO_iters'].append(REO_iters)
        results_dict['GPO_iters'].append(GPO_iters)

        if REO_error < 0.01:
            results_dict['num_REO_correct'] += 1
        if GPO_error < 0.01:
            results_dict['num_GPO_correct'] += 1

        # if False: #comb_error > 1:
        #     print( "REO error = ", REO_error)
        #     print( "GPO_error = ", GPO_error)
        #     print( "diff = ", diff_error)
        #     print( "REO_iters = ", REO_iters)
        #     print( "GPO_iters = ", GPO_iters)
        #     print( "initial_error = ", initial_error)
        #
        #     plt.figure(1)
        #     plt.clf()
        #     plt.plot(GPO_optimized[1, :], GPO_optimized[0, :], label='GPO')
        #     plt.plot(REO_optimized[1, :], REO_optimized[0, :], label='REO')
        #     # plt.plot(comb_optimized[1, :], comb_optimized[0, :], label='comb')
        #     plt.plot(global_state[i, 1, :], global_state[i, 0, :], label='init')
        #     plt.plot(truth[1, :], truth[0, :], label="truth")
        #     plt.legend()
        #     plt.show()

    return results_dict

if __name__ == '__main__':
    subprocess.Popen(['mkdir', '-p', 'tests/well_conditioned/plots'])

    cwd = os.getcwd()
    os.chdir("tests/well_conditioned")

    generate_house("data.pkl", 0, 100)
    print( "running optimization")
    results = run("data.pkl")

    results['avg_REO_error'] = sum(results['REO_errors']) / float(results['num_robots'])
    results['avg_GPO_error'] = sum(results['GPO_errors']) / float(results['num_robots'])
    results['avg_REO_iter'] = float(sum(results['REO_iters'])) / float(results['num_robots'])
    results['avg_GPO_iter'] = float(sum(results['GPO_iters'])) / float(results['num_robots'])
    results['max_REO_error'] = max(results['REO_errors'])
    results['max_GPO_error'] = max(results['GPO_errors'])


    # Plot Error Histogram
    hist_options = {"edgecolor":'black', "linewidth":0.5}
    plt.figure(1, figsize=(12,8))
    plt.set_cmap('Set2')
    plt.subplot(2,2,1)
    plt.hist(results['REO_errors'], label="REO", **hist_options)
    plt.legend()
    plt.subplot(2,2,3)
    plt.hist(results['GPO_errors'], label="GPO", **hist_options)
    plt.legend()
    plt.subplot(1,2,2)
    plt.hist(results['diff_errors'], label="REO-GPO", **hist_options)
    plt.legend()
    plt.savefig("plots/error_hist" + str(results['num_robots']) + ".png")

    # Plot Iterations Histogram
    plt.figure(1, figsize=(12,8))
    plt.set_cmap('Set2')
    plt.subplot(2,1,1)
    plt.hist(results['REO_iters'], label="REO", **hist_options)
    plt.legend()
    plt.subplot(2,1,2)
    plt.hist(results['GPO_iters'], label="GPO", **hist_options)
    plt.legend()
    plt.savefig("plots/iter_hist" + str(results['num_robots']) + ".png")

    # Plot all the trajectories
    print( "plotting trajectories")
    plt.figure(2, figsize=(12,9))
    plt.set_cmap('Set1')
    for j, (REO, GPO, nodes, truth) in tqdm(enumerate(zip(results['REO_opt'], results['GPO_opt'], results['nodes'], results['truth'])), total=results['num_robots']):
        initial_pos = np.array([[nodes[i][1], nodes[i][2]] for i in range(len(nodes))])
        plt.clf()
        ax=plt.subplot(111)

        plt.plot(initial_pos[:,0], initial_pos[:,1], label='initial', linewidth=3, dashes=[10, 5], alpha = 0.5, color='g')
        plt.plot(truth[0,:], truth[1,:], label="truth", linewidth=1, alpha=1, color='k')
        plt.plot(REO[0,:], REO[1,:], label='REO', linewidth=3,  alpha=0.8, dashes=[4,2], color='b')
        plt.plot(GPO[0, :], GPO[1, :], label='GPO', linewidth=3, alpha=0.8, dashes=[2, 4], color='r')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False, shadow=False, ncol=4)
        plt.savefig("plots/traj" + str(j).zfill(3) + ".svg")
        if j > 100:
            break



    pass

    # REO_errors = []
    # REO_iters = []
    # GPO_errors = []
    # comb_errors = []
    # for angle in tqdm(np.arange(0., 360., 0.5)):
    #     # print( angle)
    #     generate_house(0., 1000)
    #     results = run()
    #     REO_errors.append(results['REO_errors'])
    #     GPO_errors.append(results['GPO_errors'])
    #     REO_iters.append(results['avg_REO_iter'])
    #     comb_errors.append(results['comb_errors'])
    #
    # REO_errors = np.array(REO_errors)
    # GPO_errors = np.array(GPO_errors)
    # comb_errors = np.array(comb_errors)
    #
    # plt.figure(3)
    # for i in range(GPO_errors.shape[1]):
    #     if i == 0:
    #         plt.plot(np.arange(0., 360., 0.5), GPO_errors[:,i], 'c.', label="GPO", alpha=0.5)
    #         # plt.plot(np.arange(0., 360., 0.5), REO_errors[:,i], 'b.', label="REO", alpha=0.5)
    #         plt.plot(np.arange(0., 360., 0.5), comb_errors[:, i], 'm.', label="REO/GPO", alpha=0.5)
    #     else:
    #         plt.plot(np.arange(0., 360., 0.5), GPO_errors[:,i], 'c.', alpha=0.5)
    #         # plt.plot(np.arange(0., 360., 0.5), REO_errors[:,i], 'b.', alpha=0.5)
    #         plt.plot(np.arange(0., 360., 0.5), comb_errors[:, i], 'm.', alpha=0.5)
    # plt.legend()
    # plt.xlabel("angle offset (deg)")
    # plt.show()
