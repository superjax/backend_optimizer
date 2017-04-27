from backend import *
from robot import *
from controller import *
from tqdm import tqdm


if __name__ == "__main__":
    dt = 0.1
    time = np.arange(0, 5000.01, dt)

    robots = []
    controllers = []
    num_robots = 1
    KF_frequency_s = 1.0

    map = Backend("Noisy Map")
    true_map = Backend("True Map")

    start_pose_range = [15, 15, 2]

    start_poses = [[randint(-start_pose_range[0], start_pose_range[0])*10,
                   randint(-start_pose_range[1], start_pose_range[1])*10,
                   randint(-start_pose_range[2], start_pose_range[2])*pi/2] for r in range(num_robots)]
    start_poses[0] = [0, 0, 0]

    P_perfect = np.array([[0.00001, 0, 0], [0, 0.00001, 0], [0, 0, 0.00001]])
    G = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.1]])
    print("simulating robots")

    controllers = [Controller(start_poses[r]) for r in range(num_robots)]
    robots = [Robot(r, G, start_poses[r]) for r in range(num_robots)]

    for r in range(num_robots):
        map.add_agent(r, start_poses[r])


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
                map.add_odometry(e)
    map.finish_up()

    map.plot()
    plt.show()

    debug = 1