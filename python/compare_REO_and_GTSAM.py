from REO import REO, invert_transform, concatenate_transform
from backend_optimizer import backend_optimizer
from keyframe_matcher_sim import KeyframeMatcher
from MC_robot import *
from controller import *
import numpy as np
import pickle
import matplotlib.pyplot as plt

def generate_data():
    trajectory_time = 60.0
    dt = 0.01
    time = np.arange(0, trajectory_time, dt)

    KF_frequency_s = 1.0
    num_trajectories = 10

    Q = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.08]])

    truth = []
    odometry = []
    keyframes = []

    controller = Controller([0, 0, 0])
    robots = MC_Robot(Q, [0, 0, 0], num_trajectories)
    print "Generating Trajectories"

    odom = []
    for t in time:
        truth.append(robots.state())
        u = controller.control(t, robots.state())
        odom.append(robots.propagate_dynamics(u, dt))

        if t % KF_frequency_s == 0 and t > 0:
            # Declare a new keyframe
            edges, KF = robots.reset()
            odometry.append(edges)
            keyframes.append((KF, '0_'+str(robots.keyframe_id()).zfill(3)))
    f = open('MC_trajectories.pkl', 'w')
    data = dict()
    data['keyframes'] = keyframes
    data['odometry'] = odometry
    data['truth'] = truth
    pickle.dump(data, f)





def get_global_pose(edges, x0):
    x = np.zeros((edges.shape[0], edges.shape[1] + 1))
    x[:, 0] = x0
    for i in range(edges.shape[1]):
        x[:,i + 1] = concatenate_transform(x[:,i], edges[:,i])
    return x

if __name__ == '__main__':

    generate_data()
    f = open('MC_trajectories.pkl', 'r')
    data = pickle.load(f)
    odometry = np.array(data['odometry'])
    odometry = np.transpose(odometry, (2, 1, 0))
    keyframes = np.array(data['keyframes'])
    truth = np.array(data['truth'])
    num_robots = odometry.shape[0]

    # Create global state estimates
    global_state = np.zeros((num_robots, 3, odometry.shape[2] + 1))
    for robot in range(num_robots):
        global_state[robot, :, :] = get_global_pose(odometry[robot, :, :], np.zeros(3))


    # Load Keyframes into the keyframe matcher
    kf_matcher = KeyframeMatcher()
    for keyframe in keyframes:
        kf_matcher.add_keyframe(*keyframe)

    # Find loop closures
    loop_closures = kf_matcher.find_loop_closures()














