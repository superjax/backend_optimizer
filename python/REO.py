import numpy as np
import scipy.sparse
import scipy.linalg
import sys
import numpy.matlib
import tqdm


def concatenate_transform(T1, T2):
    cs = np.cos(T1[2])
    ss = np.sin(T1[2])
    x = T1[0] + T2[0] * cs - T2[1] * ss
    y = T1[1] + T2[0] * ss + T2[1] * cs
    psi = T1[2] + T2[2]
    if psi > np.pi:
        psi -= 2.*np.pi
    elif psi < -np.pi:
        psi += 2.*np.pi
    return [x, y, psi]


def invert_transform(T):
    cs = np.cos(T[2])
    ss = np.sin(T[2])
    dx = -(T[0] * cs + T[1] * ss)
    dy = -(- T[0] * ss + T[1] * cs)
    psi = -T[2]
    return [dx, dy, psi]

def invert_edges(x, dirs, indexes):
    for index in indexes:
        x[:, index] = invert_transform(x[:, index])
        dirs[index] *= -1.0

def get_global_pose(edges, x0):
    x = np.zeros((edges.shape[0], edges.shape[1] + 1))
    x[:, 0] = x0
    for i in range(edges.shape[1]):
        x[:,i + 1] = concatenate_transform(x[:,i], edges[:,i])
    return x


class REO():
    def __init__(self):
        debug = 1

    def read_g2o(self, filename):
        f = open(filename, 'r')

        edges = []
        dirs = []
        Omegas = []
        lc = []
        lc_dirs = []
        lc_Omegas = []

        for line in f:
            if "EDGE_SE2" in line:
                read_stuff = line.split()
                node_ids = map(int, [read_stuff[1], read_stuff[2]])
                # Consecutive nodes (odometry going forward)
                if node_ids[1] == node_ids[0] + 1:
                    edges.append(map(float, [read_stuff[3], read_stuff[4], read_stuff[5]]))
                    dirs.append(1)
                    Omegas.append(np.diag(map(float, [read_stuff[6], read_stuff[9], read_stuff[11]])))
                # Consecutive nodes, but backwards
                elif node_ids[1] == node_ids[0] - 1:
                    edges.append(map(float, [read_stuff[3], read_stuff[4], read_stuff[5]]))
                    dirs.append(-1)
                    Omegas.append(np.diag(map(float, [read_stuff[6], read_stuff[9], read_stuff[11]])))
                # Loop closure (forwards)
                elif node_ids[1] > node_ids[0]:
                    lc.append([map(float, [read_stuff[3], read_stuff[4], read_stuff[5]])])
                    lc_dirs.append(1)
                    lc_Omegas.append(np.diag(map(float, [read_stuff[6], read_stuff[9], read_stuff[11]])))
                # Loop closure backwards
                else:
                    lc.append([map(float, [read_stuff[3], read_stuff[4], read_stuff[5]])])
                    lc_dirs.append(-1)
                    lc_Omegas.append(np.diag(map(float, [read_stuff[6], read_stuff[9], read_stuff[11]])))
        return np.array(edges).T, dirs, Omegas, np.array(lc).T, lc_dirs, lc_Omegas

    def output_g2o(self, filename, edges, dirs, Omegas, lc, lc_dirs, lc_omegas, global_pose):
        f = open(filename, 'w')
        i = 0
        for pose in global_pose.T:
            f.write('VERTEX_SE2 %d %f %f %f\n' % (i, pose[0], pose[1], pose[2]))
            i += 1
        f.write('FIX 0\n')
        i = 0
        for edge in edges.T:
            if dirs[i] > 0:
                f.write('EDGE_SE2 %d %d %f %f %f %f %f %f %f %f %f\n' % (i, i+1, edge[0], edge[1], edge[2], Omegas[i][0][0], 0, 0, Omegas[i][1][1], 0, Omegas[i][2][2]))
            else:
                f.write('EDGE_SE2 %d %d %f %f %f %f %f %f %f %f %f\n' % (
                i+1, i, edge[0], edge[1], edge[2], Omegas[i][0][0], 0, 0, Omegas[i][1][1], 0, Omegas[i][2][2]))
            i += 1
        i = 0
        for l in lc.T:
            f.write('EDGE_SE2 %d %d %f %f %f %f %f %f %f %f %f\n' % (l[3], l[4], l[0], l[1], l[2], lc_omegas[i][0][0], 0, 0, lc_omegas[i][1][1], 0, lc_omegas[i][2][2]))
            i += 1

    def optimize(self, z_bar, dirs, Omegas, lcs, lc_omegas, lc_dirs, cycles, iters, epsilon, x0 = []):

        # create giant combined omega
        Omega = scipy.sparse.block_diag(Omegas).toarray()
        O_inv = np.linalg.inv(Omega)

        # Initialize z_hat
        if type(x0) is np.ndarray:
            z_hat = x0.copy()
        else:
            z_hat = z_bar.copy()

        for i in range(len(lc_dirs)):
            if lc_dirs[i] < 0:
                lcs[:, i] = invert_transform(lcs[:, i])

        diff = sys.float_info.max
        iter = 0

        while iter < iters and diff > epsilon:

            # How far have we deviated from our original measurements (be sure to handle pi-wrap)
            delta = z_hat - z_bar
            for psi in delta[2,:]:
                if psi > np.pi:
                    psi -= 2.0*np.pi
                elif psi <= -np.pi:
                    psi += 2.0*np.pi

            A = Omega.copy()
            b = -Omega.dot(delta.flatten(order='F'))

            # Determine in how many loops each edge appears
            edge_counter = dict()
            for loop in cycles:
                for i in range(len(loop)):
                    from_id = loop[i]
                    if i == len(loop)-1:
                        to_id = loop[0]
                    else:
                        to_id = loop[i+1]

                    edge = str(from_id).zfill(3) + "_" + str(to_id).zfill(3)
                    if edge in edge_counter:
                        edge_counter[edge] += 1.0
                    else:
                        edge_counter[edge] = 1.0

            for i in range(len(cycles)):
                this_edges = z_hat[:, cycles[i]]
                this_dirs = dirs[cycles[i]]
                this_lc = lcs[:, i]
                this_lc_omega = lc_omegas[i]
                loop = cycles[i]

                # Create the mask to put the edges in their place when we are done
                mask = np.zeros((3*z_hat.shape[1], 3*this_edges.shape[1]))
                for j in range(len(cycles[i])):
                    mask[3*cycles[i][j]:3*cycles[i][j]+3,3*j:3*j+3] = np.eye(3)

                # Find Jacobian of cycle at this linearization point
                H_az = self.calc_jacobian_of_string_with_reversed_edges(this_edges, this_dirs, loop, edge_counter)

                # concatenate edges
                z_long_way = self.compound_edges(this_edges, this_dirs)

                # Difference the edges (take care to handle the pi-wrap)
                residual = z_long_way - this_lc
                if residual[2] > np.pi:
                    residual[2] -= 2.0*np.pi
                if residual[2] <= -np.pi:
                    residual[2] += 2.0*np.pi

                # m = np.zeros((1, H_az.shape[1]))
                m = np.zeros((H_az.shape[1], H_az.shape[1]))
                for i in range(len(loop)):
                    from_id = loop[i]
                    if i == len(loop)-1:
                        to_id = loop[0]
                    else:
                        to_id = loop[i+1]

                    edge = str(from_id).zfill(3) + "_" + str(to_id).zfill(3)
                    m[3*i, 3*i] = 1.0/edge_counter[edge]
                    m[3*i+1, 3 * i + 1] = 1.0 / edge_counter[edge]
                    m[3*i+2, 3 * i + 2] = 1.0 / edge_counter[edge]


                tempA = m.dot(H_az.T.dot(this_lc_omega).dot(H_az))
                A += mask.dot(tempA).dot(mask.T)
                b -= mask.dot(m.dot(H_az.T.dot(this_lc_omega).dot(residual))).flatten()


            val = self.get_cost_function_value(z_hat, z_bar, O_inv)

            if val < .00315 and iter !=0:
                break


            # Solve
            # print("Solving")
            z_star = scipy.linalg.solve(A, b)
            diff = scipy.linalg.norm(z_star.flatten())

            z_star = z_star.reshape(z_hat.shape, order='F')

            z_hat += z_star
            # print( "REO iter: ", iter, ":", diff)

            iter += 1

        # Update estimate
        return z_hat.copy(), diff, iter


    def get_cost_function_value(self, z_hat, z_bar, Omega):
        z_hat = z_hat.flatten('F')
        z_bar = z_bar.flatten('F')

        e = z_hat - z_bar

        val = e.T.dot(Omega).dot(e)
        print(val)

        return val


    def compound_edges(self, z, dirs):
        p = np.zeros(3)
        i = 0
        for d, edge in zip(dirs, z.T):
            if d > 0:
                p = concatenate_transform(p, edge)
            else:
                p = concatenate_transform(p, invert_transform(edge))

            # print( p)

            # wrap angle to +/- pi
            if p[2] > np.pi:
                p[2] -= 2.0 * np.pi
            if p[2] <= -np.pi:
                p[2] += 2.0 * np.pi
        return p

    def calc_jacobian_of_string_with_reversed_edges(self, z, dirs, loop, edge_counter):

        dtrans = z[0:2,:]
        thetas = z[2,:]

        num_edges = len(dirs)

        H = np.zeros((3, 3*len(dirs)))

        cumsum_angle = 0
        angles = np.zeros(num_edges)
        # Calculate the rotation jacobian
        for i in range(num_edges):
            if dirs[i] > 0:
                angles[i] = cumsum_angle
                cumsum_angle += thetas[i]
            else:
                # if the edge is inverted, the angle shows up early, and it is backward
                cumsum_angle -= thetas[i]
                angles[i] = -cumsum_angle

            from_edge = str(loop[i]).zfill(3)
            if i == num_edges - 1:
                to_edge = str(loop[0]).zfill(3)
            else:
                to_edge = str(loop[i+1]).zfill(3)

            edge = from_edge + "_" + to_edge
            n = edge_counter[edge]

            # Fill in the translation jacobian
            H[0:2, 3*i:3*i+2] = self.R(angles[i])  # Multiply by n here

        for i in range(num_edges):
            # Calculate the rotation jacobian
            j = num_edges - 1
            dt_dtheta = np.zeros(2)
            while j > i:
                dt_dtheta += self.R(angles[j] + np.pi/2.0).dot(dtrans[:,j])
                j -= 1
            if dirs[i] < 0:
                dt_dtheta += self.R(angles[i] + np.pi/2.0).dot(dtrans[:,i])
                dt_dtheta *= -1.0
            H[0:2, 3*i+2] = dt_dtheta  # Multiply by n here
            H[2, 3*i+2] = dirs[i]  # Multiply by n here
        return H

    def R(self,theta):
        ct = np.cos(theta)
        st = np.sin(theta)
        return np.array([[ct, -st],
                         [st,  ct]])


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    optimizer = REO()

    read_file = False
    if read_file:
        edges, dirs, Omegas, lcs, lc_dirs, lc_Omegas = optimizer.read_g2o("/home/superjax/Code/MMO_proposal/matlab/N30/house10.g2o")
        lc = lcs[:,0]
        lc_dir = lc_dirs[0]
        lc_omega = lc_Omegas[0]

    else:
        perfect_edges = np.array([[1., 1., 1., 1., 2**0.5, 2**0.5/2.0, 2**0.5/2.0, 2**0.5],
                                  [0., 0., 0., 0., 0,      0,          0,          0],
                                  [np.pi/2.0, np.pi/2.0, np.pi/2.0, 3.0*np.pi/4.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, 0]])

        dirs = np.array([1, 1, 1, 1, 1, 1, 1, 1])

        optimizer.invert_edges(perfect_edges, dirs, [5, 7, 2])


        Omegas = [np.diag([10., 10., 100.]) for i in range(perfect_edges.shape[1])]

        edge_noise = np.array([[np.random.normal(0, 1./Omegas[i][0][0]) for i in range(perfect_edges.shape[1])],
                               [np.random.normal(0, 1./Omegas[i][1][1]) for i in range(perfect_edges.shape[1])],
                               [np.random.normal(0, 1./Omegas[i][2][2]) for i in range(perfect_edges.shape[1])]])

        noisy_edges = perfect_edges + edge_noise

        lc = np.array([[1.0, 0.0, 1.0, 1.0, 0.5, 1.5],
                       [0.0, 0.0, 1.0, 1.0, 1.5, 0.5],
                       [-np.pi/4.0, np.pi/4.0, 3.0*np.pi/4.0, -np.pi, -3*np.pi/4.0, 3*np.pi/4.0]])
        lc_omega = [np.diag([100, 100, 1000]) for i in range(lc.shape[1])]
        lc_dir =  np.array([1,1,1,1,1,1])

        cycles = [[i for i in range(perfect_edges.shape[1])],
                  [i for i in range(4)],
                  [i for i in range(5)],
                  [i for i in range(2)],
                  [i for i in range(6)],
                  [i+1 for i in range(5)]]

        # Turn off some loop closures
        active_lc = [0, 2]
        lc = lc[:,active_lc]
        lc_omega = [lc_omega[i] for i in active_lc]
        lc_dir = lc_dir[active_lc, None]
        cycles = [cycles[i] for i in active_lc]

        edges = perfect_edges
        edges = noisy_edges

        g2o_lc = np.zeros((5, lc.shape[1]))
        g2o_lc[0:3,:] = lc
        for i in range(lc.shape[1]):
            g2o_lc[3,i] = cycles[i][0]
            g2o_lc[4,i] = cycles[i][-1] + 1


    plt.figure(0)
    plt.clf()
    plt.title('optimization')
    num_iters = 10000
    zbar = edges.copy()
    iter = 0
    diff = 10000
    while iter < num_iters and diff > 0.00001:
        print('New iteration')

        global_pose = np.zeros((3, edges.shape[1]+1))
        for i in range(edges.shape[1]):
            if dirs[i] > 0:
                global_pose[:,i+1] = np.array(optimizer.concatenate_transform(global_pose[:,i], edges[:,i]))
            else:
                global_pose[:, i + 1] = np.array(optimizer.concatenate_transform(global_pose[:, i], optimizer.invert_transform(edges[:, i])))

        optimizer.output_g2o('simple_house.g2o', edges, dirs, Omegas, g2o_lc, lc_dir, lc_omega, global_pose)

        global_lc_location = np.zeros((3, 2 * lc.shape[1]))
        for i in range(lc.shape[1]):
            global_lc_location[:,i*2] = global_pose[:,cycles[i][0]]
            global_lc_location[:,i*2+1] = np.array(optimizer.concatenate_transform(global_pose[:,cycles[i][0]], lc[:,i]))

        plt.plot(global_pose[1,:], global_pose[0,:], 'b', alpha=0.1+(float(iter)/float(num_iters)**1.5)*0.9)
        for i in range(lc.shape[1]):
            plt.plot(global_lc_location[1, i*2:i*2+2], global_lc_location[0, i*2:i*2+2], 'r')

        debug = 1

        cycle = [cycles[np.random.randint(0, len(cycles))]]

        edges, diff = optimizer.optimize(zbar, dirs, Omegas, lc, lc_omega, lc_dir, cycles, 1, 0.00001, edges, SGD=False)
        iter += 1
        # print( "error = ", diff, "iters =", iter)
    plt.show()

def REO_opt(edges, nodes, origin_node, num_iters, tol):

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
            odom.append([edge[2], edge[3], edge[4]])
            Omegas.append(np.diag(np.array([edge[5], edge[6], edge[7]])))
            if to_id == from_id + 1: # forwards
                dirs.append(1)
            elif to_id == from_id - 1: # backwards
                dirs.append(-1)

        # Loop closure
        else:
            lc.append([edge[2], edge[3], edge[4]])
            try:
                lc_omegas.append(np.diag([edge[5], edge[6], edge[7]]))
            except:
                debug = 1
            if to_id > from_id: # forwards
                lc_dirs.append(1)
                cycles.append([i + from_id for i in range(to_id - from_id)])
            else: # backwards
                cycles.append([i + to_id for i in range(from_id - to_id)])
                lc_dirs.append(-1)

    z_hat, diff, iters = reo.optimize(np.array(odom).T, np.array(dirs), Omegas, np.atleast_2d(np.array(lc).squeeze()).T,
                                      lc_omegas, np.array(lc_dirs), np.array(cycles), num_iters, tol, x0=nodes)
    x_hat = get_global_pose(z_hat, np.array([0, 0, 0]))
    return x_hat, iters
