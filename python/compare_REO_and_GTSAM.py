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

def generate_house():
    num_robots = 1000
    perfect_edges = np.array([[1., 1., 1., 1., 2 ** 0.5, 2 ** 0.5 / 2.0, 2 ** 0.5 / 2.0, 2 ** 0.5],
                              [0., 0., 0., 0., 0, 0, 0, 0],
                              [np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, 3.0 * np.pi / 4.0, np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, 0]])

    x0 = np.array([0, 0, 0])
    x0_rot = np.array([0, 0, np.pi])

    dirs = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    Omegas = [np.diag([1e2, 1e2, 1e3]) for i in range(perfect_edges.shape[1])]
    # Omegas = [np.diag([1e2, 1e2, 1e3]) for i in range(perfect_edges.shape[1])]

    odometry = np.zeros((num_robots, 3, perfect_edges.shape[1]))
    global_estimate = np.zeros((num_robots, 3, perfect_edges.shape[1]+1))
    truth = get_global_pose(perfect_edges, x0.copy())
    for robot in range(num_robots):
        edge_noise = np.array([[np.random.normal(0, 1. / Omegas[i][0][0]) for i in range(perfect_edges.shape[1])],
                               [np.random.normal(0, 1. / Omegas[i][1][1]) for i in range(perfect_edges.shape[1])],
                               [np.random.normal(0, 1. / Omegas[i][2][2]) for i in range(perfect_edges.shape[1])]])

        noisy_edges = perfect_edges + edge_noise
        odometry[robot,:,:] = noisy_edges + edge_noise
        global_estimate[robot,:,:] = get_global_pose(noisy_edges, x0.copy())

    # invert_edges(perfect_edges, dirs, [5, 7, 2])

    lc = np.array([[1.0, 1.0, 0.5, 0.0, 0.0,      0., 0.5, 2.**0.5/2., 1.0],
                   [0.0, 1.0, 1.5, 1.0, 1.0,      1., -0.5, -2.**0.5/2., 1.0],
                   [-np.pi / 4.0, 3.*np.pi / 4.0, -3.*np.pi / 4.0, -np.pi/2.0, -np.pi / 4.0,       -3.*np.pi/4., np.pi/4., -np.pi/2., -3.*np.pi/4.]])
    lc_omega = [np.diag([1e5, 1e5, 1e5]) for i in range(lc.shape[1])]
    lc_dir = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])

    cycles = [['0_'+str(i).zfill(3) for i in range(perfect_edges.shape[1])],
              ['0_'+str(i).zfill(3) for i in range(5)],
              ['0_'+str(i).zfill(3) for i in range(6)],
              ['0_'+str(i).zfill(3) for i in range(3)],
              ['0_'+str(i).zfill(3) for i in range(7)],

              ['0_' + str(i+3).zfill(3) for i in range(2)],
              ['0_' + str(i+2).zfill(3) for i in range(4)],
              ['0_' + str(i+4).zfill(3) for i in range(4)],
              ['0_' + str(i+1).zfill(3) for i in range(6)]]

    # Turn off some loop closures
    active_lc = [0,1]
    lc = lc[:, active_lc]
    lc_omega = [lc_omega[i] for i in active_lc]
    lc_dir = lc_dir[active_lc, None]
    cycles = [cycles[i] for i in active_lc]

    # edges = perfect_edges
    # edges = noisy_edges

    f = open('MC_trajectories.pkl', 'wb')
    data = dict()
    data['odometry'] = odometry
    data['global_state'] = global_estimate
    data['truth'] = truth.T
    data['Q'] = scipy.linalg.inv(Omegas[0])

    # plt.figure(5)
    # plt.plot(truth[1,:], truth[0,:])
    # plt.plot(global_estimate[1,:], global_estimate[0,:])
    # plt.show()

    loop_closures = []
    for i in range(lc.shape[1]):
        new_lc = dict()
        new_lc['covariance'] = scipy.linalg.inv(lc_omega[i].tolist())
        if lc_dir[i] > 0:
            new_lc['from_node_id'] = cycles[i][0]
            new_lc['to_node_id'] = cycles[i][-1]
        else:
            new_lc['from_node_id'] = cycles[i][-1]
            new_lc['to_node_id'] = cycles[i][0]
        new_lc['transform'] = lc[:,i].tolist()
        loop_closures.append(new_lc)
    data['loop_closures'] = loop_closures

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

def GPO_opt(edge_names, odom, lcs, gst, cov):
    edge_lists = list(edge_names)

    # Optimize with Global Pose Optimization
    GPO = backend_optimizer.Optimizer()
    GPO.new_graph('0_000', 0)
    edge_lists.extend(odom.tolist())
    edge_lists.extend([[cov[0][0] for j in range(odom.shape[1])]])
    edge_lists.extend([[cov[1][1] for j in range(odom.shape[1])]])
    edge_lists.extend([[cov[2][2] for j in range(odom.shape[1])]])
    for lc in lcs:
        edge_lists[0].append(lc['from_node_id'])
        edge_lists[1].append(lc['to_node_id'])
        edge_lists[2].append(lc['transform'][0])
        edge_lists[3].append(lc['transform'][1])
        edge_lists[4].append(lc['transform'][2])
        edge_lists[5].append(lc['covariance'][0][0])
        edge_lists[6].append(lc['covariance'][1][1])
        edge_lists[7].append(lc['covariance'][2][2])

    node_lists = [node_names[1:]]
    node_lists.extend(gst[:, 1:].tolist())

    nodes_transpose = [list(j) for j in zip(*node_lists)]
    edges_transpose = [list(j) for j in zip(*edge_lists)]

    GPO.add_edge_batch(nodes_transpose, edges_transpose)
    for i in range(10):
        GPO.optimize()
    out_dict = GPO.get_optimized()
    opt_pose = []
    for node in out_dict['nodes']:
        opt_pose.append(node[1:])
    opt_pose = np.array(opt_pose)
    return opt_pose.T.copy()

def REO_opt(edges, odom, loops, gst, cov):

    reo = REO()
    dirs = np.ones(odom.shape[1])
    Qinv = scipy.linalg.inv(cov)
    Omegas = [Qinv for i in range(odom.shape[1])]

    lc_dirs = []
    lcs = np.zeros((3, len(loops)))
    lc_omegas = []
    cycles = []
    for i in range(len(loops)):
        lc_dirs.append(1)
        lc_omegas.append(scipy.linalg.inv(loops[i]['covariance']))
        lcs[:,i] = np.array(loops[i]['transform'])
        from_id = int(loops[i]['from_node_id'].split('_')[1])
        to_id = int(loops[i]['to_node_id'].split('_')[1])
        if to_id > from_id:
            cycle = range(from_id, to_id+1)
        else:
            cycle = range(to_id, from_id+1)
        cycles.append(cycle)
    z_hat, diff, iter = reo.optimize(odom, dirs, Omegas, lcs, lc_omegas, lc_dirs, cycles, 100, 1e-8)
    x_hat = get_global_pose(z_hat, np.array([0, 0, 0]))
    return x_hat, iter


if __name__ == '__main__':

    # generate_data()
    generate_house()
    f = open('MC_trajectories.pkl', 'rb')
    data = pickle.load(f)
    odometry = data['odometry']
    global_state = data['global_state']
    # keyframes = np.array(data['keyframes'])
    truth = np.array(data['truth'])
    loop_closures = data['loop_closures']
    Q = data['Q']

    # Add loop closure at the end
    # final_lc = {'from_node_id': '0_000',
    #             'to_node_id': '0_' + str(global_state.shape[0] - 1).zfill(3),
    #             'transform': truth[-1,:],
    #             'covariance': np.eye(3)*1e-9}
    # loop_closures.append(final_lc)

    num_robots = odometry.shape[0]



    node_names = ['0_' + str(i).zfill(3) for i in range(global_state.shape[2])]

    # Odometry edges
    edges = [node_names[0:-1], node_names[1:]]

    # For each agent, optimize with REO and GTSAM
    REO_error_list = []
    GPO_error_list = []
    diff_error_list = []
    REO_avg_iter_sum = 0.
    error_threshold = 15.
    REO_correct_count = 0
    GPO_correct_count = 0
    for i in tqdm(range(num_robots)):
        # truth_optimized = GPO_opt(edges, odometry[i, :, :], loop_closures, truth.T, Q)
        GPO_optimized = GPO_opt(edges, odometry[i, :, :].copy(), loop_closures, global_state[i,:,:].copy(), Q)
        REO_optimized, REO_iters = REO_opt(edges, odometry[i, :, :].copy(), loop_closures, global_state[i,:,:].copy(), Q)

        # initial_error = scipy.linalg.norm(global_state - truth.T)
        REO_error = np.sum(scipy.linalg.norm(REO_optimized[0:2,:] - truth.T[0:2,:], axis=0))
        diff_error = np.sum(scipy.linalg.norm(REO_optimized[0:2, :] - GPO_optimized[0:2, :], axis=0))
        GPO_error = np.sum(scipy.linalg.norm(GPO_optimized[0:2,:] - truth.T[0:2,:], axis=0))

        # print "REO:", REO_error, "GPO:", GPO_error

        REO_error_list.append(REO_error)
        GPO_error_list.append(GPO_error)
        diff_error_list.append(diff_error)
        REO_avg_iter_sum += float(REO_iters)

        if REO_error < 0.5:
            REO_correct_count += 1
        if GPO_error < 0.5:
            GPO_correct_count += 1
        if REO_error > 1 or GPO_error > 1:
            print "REO error = ", REO_error
            print "GPO_error = ", GPO_error
            plt.figure(1)
            plt.clf()
            plt.plot(GPO_optimized[1,:], GPO_optimized[0,:], label='GPO')
            plt.plot(REO_optimized[1, :], REO_optimized[0,:], label='REO')
            plt.plot(global_state[i, 1, :], global_state[i, 0, :], label='init')
            # plt.plot(truth_optimized[1,:], truth_optimized[0,:], label="truth")
            plt.plot(truth.T[1, :], truth.T[0, :], label="truth")
            plt.legend()
            # plt.ion()
            plt.show()
            # plt.pause(0.001)
            # plt.ioff()
            if REO_error > 1:
                debug = 1

    print "avg REO error:", sum(REO_error_list)/float(num_robots)
    print "avg GPO error:", sum(GPO_error_list) / float(num_robots)
    print "avg REO iter:", REO_avg_iter_sum / float(num_robots)
    print "num REO correct:", REO_correct_count
    print "num GPO correct:", GPO_correct_count

    plt.figure(1)
    plt.clf()
    plt.subplot(122)
    plt.title("GPO - REO RMS error")
    plt.hist(diff_error_list, 50, normed=1, facecolor="red", alpha=0.5)
    plt.subplot(221)
    plt.title("REO RMS error")
    plt.hist(REO_error_list, 100, normed=1, facecolor="blue", alpha=0.5)
    plt.subplot(223)
    plt.title("GPO RMS error")
    plt.hist(GPO_error_list, 100, normed=1, facecolor="green", alpha=0.5)

    plt.show()
















