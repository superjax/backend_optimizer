from REO import REO, invert_transform, concatenate_transform
from backend_optimizer import backend_optimizer
from keyframe_matcher_sim import KeyframeMatcher
from MC_robot import *
from controller import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_data():
    trajectory_time = 60.0
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

    plt.figure(1)
    for i in range(num_trajectories):
        plt.plot(global_state[:,1,i], global_state[:,0, i], alpha=0.25)
    plt.plot(truth[:, 1], truth[:, 0], label="truth",  linewidth=2.0)
    plt.legend()
    plt.show()


    f = open('MC_trajectories.pkl', 'w')
    data = dict()
    data['keyframes'] = keyframes
    data['odometry'] = np.array(odometry)
    data['global_state'] = global_state
    data['truth'] = truth

    # Load Keyframes into the keyframe matcher and find loop closures
    kf_matcher = KeyframeMatcher()
    for keyframe in keyframes:
        kf_matcher.add_keyframe(*keyframe)
    loop_closures = kf_matcher.find_loop_closures()
    data['loop_closures'] = loop_closures

    pickle.dump(data, f)





def get_global_pose(edges, x0):
    x = np.zeros((edges.shape[0], edges.shape[1] + 1))
    x[:, 0] = x0
    for i in range(edges.shape[1]):
        x[:,i + 1] = concatenate_transform(x[:,i], edges[:,i])
    return x

if __name__ == '__main__':

    # generate_data()
    f = open('MC_trajectories.pkl', 'r')
    data = pickle.load(f)
    odometry = data['odometry']
    global_state = data['global_state']
    keyframes = np.array(data['keyframes'])
    truth = np.array(data['truth'])
    loop_closures = np.array(data['loop_closures'])

    num_robots = odometry.shape[0]

    # re-arrange these arrays to make things convenient
    odometry = np.transpose(odometry, (2, 1, 0))
    global_state = np.transpose(global_state, (2, 1, 0))

    node_names = ['0_' + str(i).zfill(3) for i in range(global_state.shape[0] + 1)]

    edges = [node_names[0:-1], node_names[1:]]

    # For each agent, optimize with REO and GTSAM
    for i in range(num_robots):

        edge_lists = edges

        # Optimize with Global Pose Optimization
        GPO = backend_optimizer.Optimizer()
        GPO.new_graph('0_000', 0)
        edge_lists.extend(odometry[:,:,i].T.tolist())
        edges.extend([])
















