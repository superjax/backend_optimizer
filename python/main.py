from keyframe_matcher_sim import KeyframeMatcher
from robot import *
from controller import *
from tqdm import tqdm
import os
import time


if __name__ == "__main__":
    stamp = time.time()
    if not os.path.isdir("tests/well_conditioned/demo_2_plots"):
        os.mkdir("tests/well_conditioned/demo_2_plots")
    os.chdir("tests/well_conditioned")
    dt = 0.1    
    time = np.arange(0, 300.01, dt)
    colors = ['r', 'b', 'g', 'k', 'm']

    robots = []
    controllers = []
    num_robots = 1  # Will up to 10 later. Start with 2 while familiarizing myself.
    KF_frequency_s = 0.5
    plot_frequency_s = 10

    start_pose_range = [5, 3, 2]

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

    for t in tqdm(time):
        for r in range(num_robots):
            # Run each robot through the trajectory
            u = controllers[r].control(t, robots[r].state(), [55, 35])
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

    # Plot the noisy initial data
    plt.figure(1)
    for r in range(num_robots):
        plt.plot(np.array(robots[r].plt_nodes)[:, 1], np.array(robots[r].plt_nodes)[:, 2], colors[r % num_robots])

    # Plot the true initial data
    plt.figure(2)
    for r in range(num_robots):
        plt.plot(np.array(robots[r].true_nodes)[:, 1], np.array(robots[r].true_nodes)[:, 2], colors[r % num_robots])

    # Plot the true map
    plt.figure(3)
    for r in range(num_robots):
        plt.plot(np.array(robots[r].I_nodes)[:, 1], np.array(robots[r].I_nodes)[:, 2], colors[r%num_robots])

    plt.figure(4)
    for r in range(num_robots):
        plt.plot(np.array(robots[r].I)[:, 1], np.array(robots[r].I)[:, 2], colors[r % num_robots])
    plt.show()

    debug = 1