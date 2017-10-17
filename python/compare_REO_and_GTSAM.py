from REO import REO, invert_transform, concatenate_transform
from backend_optimizer import backend_optimizer
from keyframe_matcher_sim import KeyframeMatcher
from MC_robot import *
from controller import *
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.linalg
from REO import invert_transform, concatenate_transform, invert_edges

def get_global_pose(edges, x0):
    x = np.zeros((edges.shape[0], edges.shape[1] + 1))
    x[:, 0] = x0
    for i in range(edges.shape[1]):
        x[:,i + 1] = concatenate_transform(x[:,i], edges[:,i])
    return x

def generate_house(angle_offset = 0, num_robots = 1000):
    print "generating", num_robots, "house trajectories"
    perfect_edges = np.array([[1., 1., 1., 1., 2 ** 0.5, 2 ** 0.5 / 2.0, 2 ** 0.5 / 2.0, 2 ** 0.5],
                              [0., 0., 0., 0., 0, 0, 0, 0],
                              [np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, 3.0 * np.pi / 4.0, np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, 0]])

    x0 = np.array([0, 0, 0])

    dirs = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    Omegas = [np.diag([1e5, 1e5, 1e4]) for i in range(perfect_edges.shape[1])]

    odometry = np.zeros((num_robots, 3, perfect_edges.shape[1]))
    global_estimate = np.zeros((num_robots, 3, perfect_edges.shape[1]+1))
    truth = get_global_pose(perfect_edges, x0.copy())
    # invert_edges(perfect_edges, dirs, [5, 7, 2])

    cycles = [[0, 8],
              [0, 5],
              [0, 6],
              [0, 3],
              [0, 7],
              [3, 5],
              [2, 6],
              [4, 8],
              [1, 7]]

    lc = np.zeros([3, len(cycles)])
    for i, cycle in enumerate(cycles):
        for j in range(cycle[0], cycle[1]):
            lc[:,i] = concatenate_transform(lc[:,i], perfect_edges[:,j])

    lc_omega = [np.diag([1e10, 1e10, 1e10]) for i in range(lc.shape[1])]

    # Turn off some loop closures
    active_lc = [0, 1, 2, 4, 5, 7, 8]
    lc = lc[:, active_lc]
    lc_omega = [lc_omega[i] for i in active_lc]
    cycles = [cycles[i] for i in active_lc]

    edge_g2o_lists = []
    nodes_g2o_lists = []

    for robot in tqdm(range(num_robots)):
        edge_noise = np.array([[0.*np.random.normal(0, (1. / Omegas[i][0][0])**0.5) for i in range(perfect_edges.shape[1])],
                               [0.*np.random.normal(0, (1. / Omegas[i][1][1])**0.5) for i in range(perfect_edges.shape[1])],
                               [np.random.normal(0, (1. / Omegas[i][2][2])**0.5) for i in range(perfect_edges.shape[1])]])
        edge_noise[:,0] = np.zeros(3)

        noisy_edges = perfect_edges + edge_noise
        odometry[robot,:,:] = noisy_edges + edge_noise
        global_estimate[robot,:,:] = get_global_pose(noisy_edges, x0.copy())

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

    f = open('MC_trajectories.pkl', 'wb')
    data = dict()
    data['edges'] = edge_g2o_lists
    data['nodes'] = nodes_g2o_lists
    data['global_state'] = global_estimate
    data['truth'] = truth
    pickle.dump(data, f)


def generate_data():
    trajectory_time = 600.1
    dt = 0.01
    time = np.arange(0, trajectory_time, dt)

    KF_frequency_s = 1.0
    num_trajectories = 100

    Q = np.array([[5e-2, 0, 0], [0, 5e-2, 0], [0, 0, 2e-1]])

    truth = []
    odometry = []
    keyframes = []

    controller = Controller([0, 0, 0])
    robots = MC_Robot(Q, [0, 0, 0], num_trajectories)
    print "Generating Trajectories"

    global_state = []
    global_state.append(np.zeros((3, num_trajectories)))
    for t in tqdm(time):
        truth.append(robots.state())
        u = controller.control(t, robots.state())
        odom = robots.propagate_dynamics(u, dt)

        if t % KF_frequency_s == 0 and t > 0:
            # Declare a new keyframe
            edges, KF = robots.reset()
            odometry.append(edges)
            keyframes.append((KF, '0_'+str(robots.keyframe_id()).zfill(3)))

            g_odom = np.zeros((3, num_trajectories))
            for i in range(num_trajectories):
                g_odom[:, i] = concatenate_transform(global_state[-1][:, i], edges[:,i])
            global_state.append(g_odom)

    global_state = np.array(global_state)
    truth = np.array(truth)

    # re-arrange these arrays to make things convenient
    odometry = np.array(odometry)
    odometry = np.transpose(odometry, (2, 1, 0))
    global_state = np.transpose(global_state, (2, 1, 0))

    plt.figure(1)
    for i in range(num_trajectories):
        plt.plot(global_state[:,1,i], global_state[:,0, i], alpha=0.25)
    plt.plot(truth[:, 1], truth[:, 0], label="truth",  linewidth=2.0)
    plt.legend()
    plt.show()


    f = open('MC_trajectories.pkl', 'w')
    data = dict()
    data['keyframes'] = keyframes
    data['odometry'] = odometry
    data['global_state'] = global_state
    data['truth'] = truth
    data['Q'] = Q

    # Load Keyframes into the keyframe matcher and find loop closures
    kf_matcher = KeyframeMatcher()
    for keyframe in keyframes:
        kf_matcher.add_keyframe(*keyframe)
    loop_closures = kf_matcher.find_loop_closures()
    data['loop_closures'] = loop_closures

    pickle.dump(data, f)

def combined_opt(edges, nodes, origin_node, num_iters, tol):
    # Single Iteration of REO
    x_hat_REO, iter_REO = REO_opt(edges, nodes, origin_node, num_iters, tol)

    # repackage nodes
    nodes_new = [ [nodes[i][0], x_hat_REO[0, i], x_hat_REO[1, i], x_hat_REO[2, i]] for i in xrange(len(nodes)) ]

    # Converge with GPO
    x_hat, iters = GPO_opt(edges, nodes_new, origin_node, num_iters, tol)
    # x_hat = x_hat_REO
    # iters = iter_REO
    return x_hat, iters

def GPO_opt(edges, nodes, origin_node, num_iters, tol):
    GPO = backend_optimizer.Optimizer()
    out_dict = GPO.batch_optimize(nodes, edges, origin_node, num_iters, tol)
    opt_pose = []
    for node in out_dict['nodes']:
        opt_pose.append(node[1:])
    opt_pose = np.array(opt_pose)
    return opt_pose.T.copy(), out_dict['iter']

def REO_opt(edges, nodes, origin_node, num_iters, tol, SGD=False, SGD_rate = 0):

    reo = REO()

    odom = []
    dirs = []
    Omegas = []
    lc = []
    lc_dirs = []
    lc_omegas = []
    cycles = []

    # process the input
    for edge in edges:
        from_id = int(edge[0].split("_")[1])
        to_id = int(edge[1].split("_")[1])
        # Consecutive nodes
        if abs(to_id - from_id) == 1:
            odom.append(map(float, [edge[2], edge[3], edge[4]]))
            Omegas.append(np.diag(map(float, [edge[5], edge[6], edge[7]])))
            if to_id == from_id + 1: # forwards
                dirs.append(1)
            elif to_id == from_id - 1: # backwards
                dirs.append(-1)

        # Loop closure (forwards)
        else:
            lc.append([map(float, [edge[2], edge[3], edge[4]])])
            lc_omegas.append(np.diag(map(float, [edge[5], edge[6], edge[7]])))
            if to_id > from_id:
                lc_dirs.append(1)
                cycles.append([i + from_id for i in range(to_id - from_id)])
            else:
                cycles.append([i + to_id for i in range(from_id - to_id)])
                lc_dirs.append(-1)

    z_hat, diff, iters = reo.optimize(np.array(odom).T, np.array(dirs), Omegas, np.atleast_2d(np.array(lc).squeeze()).T, lc_omegas,
                                      np.array(lc_dirs), np.array(cycles), num_iters, tol, SGD=SGD, SGD_rate = 0)
    x_hat = get_global_pose(z_hat, np.array([0, 0, 0]))
    return x_hat, iters

def run():
    f = open('MC_trajectories.pkl', 'rb')
    data = pickle.load(f)
    edges = data['edges']
    nodes = data['nodes']
    global_state = data['global_state']
    truth = np.array(data['truth'])

    num_robots = len(edges)

    # For each agent, optimize with REO and GTSAM
    REO_error_list = []
    GPO_error_list = []
    diff_error_list = []
    comb_error_list = []
    REO_avg_iter_sum = 0.
    GPO_avg_iter_sum = 0.
    comb_avg_iter_sum = 0.
    REO_correct_count = 0
    GPO_correct_count = 0
    comb_correct_count = 0
    REO_SGD_error_lists = dict()
    REO_SGD_iters = dict()
    for i in xrange(6):
        REO_SGD_error_lists[i] = []
        REO_SGD_iters[i] = []

    for i in tqdm(range(num_robots)):
        # Optimize with both optimizers
        # comb_optimized, comb_iters = combined_opt(edges[i], nodes[i], '0_000', 100, 1e-12)
        # GPO_optimized, GPO_iters = GPO_opt(edges[i], nodes[i], '0_000', 100, 1e-12)
        # REO_optimized, REO_iters = REO_opt(edges[i], nodes[i], '0_000', 100, 1e-12)
        for i in xrange(6):
            SGD_optimized, SGD_iters = REO_opt(edges[i], nodes[i], '0_000', 100, 1e-12, SGD=True, SGD_rate=i/100.)
            REO_SGD_error_lists[i].append(np.sum(scipy.linalg.norm(SGD_optimized[0:2, :] - truth[0:2, :], axis=0)))
            REO_SGD_iters[i].append(SGD_iters)

        # Calculate Error
        # initial_error = np.sum(scipy.linalg.norm(global_state[i, 0:2, :] - truth[0:2, :], axis=0))
        # REO_error = np.sum(scipy.linalg.norm(REO_optimized[0:2, :] - truth[0:2, :], axis=0))
        # diff_error = np.sum(scipy.linalg.norm(REO_optimized[0:2, :] - GPO_optimized[0:2, :], axis=0))
        # GPO_error = np.sum(scipy.linalg.norm(GPO_optimized[0:2, :] - truth[0:2, :], axis=0))
        # comb_error = np.sum(scipy.linalg.norm(comb_optimized[0:2, :] - truth[0:2, :], axis=0))

        # print "REO:", REO_error, "GPO:", GPO_error

        # REO_error_list.append(REO_error)
        # GPO_error_list.append(GPO_error)
        # diff_error_list.append(diff_error)
        # comb_error_list.append(comb_error)
        # REO_avg_iter_sum  += float(REO_iters)
        # GPO_avg_iter_sum += float(GPO_iters)
        # comb_avg_iter_sum += float(comb_iters)


        # if REO_error < 0.01:
        #     REO_correct_count += 1
        # if GPO_error < 0.01:
        #     GPO_correct_count += 1
        # if comb_error < 0.01:
        #     comb_correct_count += 1

        if False: #comb_error > 1:
            print "REO error = ", REO_error
            print "GPO_error = ", GPO_error
            # print "comb_error = ", comb_error
            print "REO_iters = ", REO_iters
            print "GPO_iters = ", GPO_iters
            # print "comb_iters = ", comb_iters
            print "initial_error = ", initial_error

            plt.figure(1)
            plt.clf()
            plt.plot(GPO_optimized[1, :], GPO_optimized[0, :], label='GPO')
            plt.plot(REO_optimized[1, :], REO_optimized[0, :], label='REO')
            # plt.plot(comb_optimized[1, :], comb_optimized[0, :], label='comb')
            plt.plot(global_state[i, 1, :], global_state[i, 0, :], label='init')
            plt.plot(truth[1, :], truth[0, :], label="truth")
            plt.legend()
            plt.show()


    # results_dict = dict()
    # results_dict['avg_REO_error'] = sum(REO_error_list) / float(num_robots)
    # results_dict['avg_GPO_error'] = sum(GPO_error_list) / float(num_robots)
    # results_dict['avg_REO_iter'] = REO_avg_iter_sum / float(num_robots)
    # results_dict['avg_GPO_iter'] = GPO_avg_iter_sum / float(num_robots)
    # results_dict['max_REO_error'] = max(REO_error_list)
    # results_dict['max_GPO_error'] = max(GPO_error_list)
    # results_dict['num_REO_correct'] = REO_correct_count
    # results_dict['num_GPO_correct'] = GPO_correct_count
    # results_dict['REO_errors'] = REO_error_list
    # results_dict['GPO_errors'] = GPO_error_list
    # results_dict['GPO_errors'] = GPO_error_list
    # # results_dict['num_comb_correct'] = comb_correct_count
    # results_dict['max_comb_error'] = max(comb_error_list)
    # results_dict['avg_comb_iter'] = comb_avg_iter_sum / float(num_robots)
    # results_dict['avg_comb_error'] = sum(comb_error_list) / float(num_robots)


    for key, item in results_dict.iteritems():
        print key, item

    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.title("REO with SGD error")
    for i in xrange(6):
        plt.hist(REO_SGD_error_lists[i*5], 100, normed=1, alpha=0.5, range=[0, 0.015], label=str(i*5)+"%")
    # plt.subplot(212)
    # plt.title("REO wihout SGD error")
    # plt.hist(REO_error_list, 100, normed=1, facecolor="green", alpha=0.5, range=[0, 0.015])


    plt.show()

    return results_dict

if __name__ == '__main__':

    generate_house(0, 100)
    print "running optimization"
    results = run()

    # REO_errors = []
    # REO_iters = []
    # GPO_errors = []
    # comb_errors = []
    # for angle in tqdm(np.arange(0., 360., 0.5)):
    #     # print angle
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


















