from backend import *
from keyframe_matcher_sim import KeyframeMatcher
from robot import *
from controller import *
from tqdm import tqdm
import os
import time


if __name__ == "__main__":
    stamp = time.time()
    os.chdir("plots")
    os.mkdir(str(stamp))
    os.chdir(str(stamp))
    dt = 0.1
    time = np.arange(0, 300.01, dt)

    robots = []
    controllers = []
    num_robots = 12
    KF_frequency_s = 0.5
    plot_frequency_s = 10

    start_pose_range = [5, 3, 2]

    start_poses = [[randint(-start_pose_range[0], start_pose_range[0])*10,
                    (-1)**r * start_pose_range[1]*10,
                    -np.pi/2.0 if r % 2 == 0 else np.pi/2.0] for r in range(num_robots)]
    # start_poses[0] = [0, 0, 0]
    # start_poses[1] = [10, 0, pi/2]
    # start_poses[2] = [10, 10, pi]
    # start_poses[3] = [0, 10, 3*pi/2]

    # start_poses[1] = [10, 0, pi/2]

    P_perfect = np.array([[0.00001, 0, 0], [0, 0.00001, 0], [0, 0, 0.00001]])
    G = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    lc_cov = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.3]])

    print("simulating robots")

    kf_matcher = KeyframeMatcher(lc_cov)
    controllers = [Controller(start_poses[r]) for r in range(num_robots)]
    robots = [Robot(r, G, start_poses[r]) for r in range(num_robots)]

    backend = Backend()

    for r in robots:
        backend.add_agent(r)

    for t in tqdm(time):
        for r in range(num_robots):
            # Run each robot through the trajectory
            u = controllers[r].control(t, robots[r].state())
            robots[r].propagate_dynamics(u, dt)
            if t % KF_frequency_s == 0 and t > 0:
                # Declare a new keyframe
                edge, KF = robots[r].reset()

                # Add odometry to all maps
                backend.add_odometry(r, str(r) + "_" + str(robots[r].keyframe_id() - 1).zfill(3),
                         str(r) + "_" + str(robots[r].keyframe_id()).zfill(3), G,
                         edge, KF)

                # Add the keyframe to the kf_matcher
                kf_matcher.add_keyframe(KF, str(r) + "_" + str(robots[r].keyframe_id()).zfill(3))

        # plot maps
        if t % plot_frequency_s == 0 and t > 0:
            # look for loop closures
            loop_closures = kf_matcher.find_loop_closures()
            for lc in loop_closures:
                backend.add_loop_closure(**lc)
            backend.plot()

    backend.finish_up()
    # backend.plot()

    print('Making movie - this make take a while')
    os.chdir('movie')
    os.system("mencoder mf://optimized*.png -mf w=800:h=600:fps=10:type=png -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o optimized.avi")
    os.system("mencoder mf://truth*.png -mf w=800:h=600:fps=10:type=png -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o truth.avi")

    plt.show()



    debug = 1
