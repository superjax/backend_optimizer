from backend import *
from robot import *
from controller import *
from tqdm import tqdm
import os


if __name__ == "__main__":
    os.chdir("movie")
    os.system('rm *.png')
    os.system('rm *.avi')
    os.chdir("..")
    dt = 0.1
    time = np.arange(0, 360.01, dt)

    robots = []
    controllers = []
    num_robots = 10
    KF_frequency_s = 1.0
    plot_frequency_s = 5

    start_pose_range = [1, 0, 2]

    start_poses = [[randint(-start_pose_range[0], start_pose_range[0])*10,
                   randint(-start_pose_range[1], start_pose_range[1])*10,
                   randint(-start_pose_range[2], start_pose_range[2])*pi/2] for r in range(num_robots)]
    start_poses[0] = [0, 0, 0]
    # start_poses[1] = [10, 0, pi/2]
    # start_poses[2] = [10, 10, pi]
    # start_poses[3] = [0, 10, 3*pi/2]

    # start_poses[1] = [10, 0, pi/2]

    P_perfect = np.array([[0.00001, 0, 0], [0, 0.00001, 0], [0, 0, 0.00001]])
    G = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01   ]])
    print("simulating robots")

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

                e = Edge(r, str(r) + "_" + str(robots[r].keyframe_id() - 1).zfill(3),
                         str(r) + "_" + str(robots[r].keyframe_id()).zfill(3), G,
                         edge, KF)

                # Add odometry to all maps
                backend.add_odometry(e)

        # plot maps
        if t % plot_frequency_s == 0 and t > 0:
            backend.plot()

    backend.finish_up()
    # backend.plot()

    print('Making movie - this make take a while')
    os.chdir('movie')
    os.system("mencoder mf://optimized*.png -mf w=800:h=600:fps=10:type=png -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o optimized.avi")
    os.system("mencoder mf://truth*.png -mf w=800:h=600:fps=10:type=png -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o truth.avi")

    plt.show()



    debug = 1