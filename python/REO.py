import numpy as np
import scipy.sparse
import scipy.linalg
import sys
import numpy.matlib

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


    def concatenate_transform(self, T1, T2):
        cs = np.cos(T1[2])
        ss = np.sin(T1[2])
        x = T1[0] + T2[0] * cs - T2[1] * ss
        y = T1[1] + T2[0] * ss + T2[1] * cs
        psi = T1[2] + T2[2]
        return [x, y, psi]

    def invert_transform(self, T):
        cs = np.cos(T[2])
        ss = np.sin(T[2])
        dx = -(T[0] * cs + T[1] * ss)
        dy = -(- T[0] * ss + T[1] * cs)
        psi = -T[2]
        return [dx, dy, psi]

    def invert_edges(self, x, dirs, indexes):
        for index in indexes:
            x[:, index] = self.invert_transform(x[:, index])
            dirs[index] *= -1.0

    # def optimize(self, z_bar, dirs, Omegas, lcs, lc_omegas, lc_dirs, cycles, iters, epsilon):
    #     edges = z_bar.copy()
    #
    #     occurrances = []
    #     for i in range(z_bar.shape[1]):
    #         occurrances.append(sum(x.count(i) for x in cycles))
    #     W = np.diag(occurrances)/float(max(occurrances))
    #     for i in range(len(cycles)):
    #         this_edges = edges[:,cycles[i]]
    #         this_edge_dirs = dirs[cycles[i]]
    #         this_edge_Omegas = [Omegas[j] for j in cycles[i]]
    #         this_lc = lcs[:, i, None]
    #         this_lc_dir = lc_dirs[i]
    #         this_lc_omega = lc_omegas[i]
    #
    #         updated_edges, error = self.optimize_cycle(this_edges, this_edge_dirs, this_edge_Omegas, this_lc,
    #                                                    this_lc_omega, this_lc_dir, 100, epsilon)
    #
    #         # only update if it's a helpful update
    #         if error < epsilon:
    #             # Weight the amount of change with the number of occurrences
    #             edge_update = np.zeros(edges.shape)
    #             edge_update[:,cycles[i]] = updated_edges
    #             edge_update = edge_update.dot(W)
    #             # update with the results of this optimization
    #             edges -= edge_update
    #     return edges

    def optimize(self, z_bar, dirs, Omegas, lcs, lc_omegas, lc_dirs, cycles, iters, epsilon, x0 = []):

        # create giant combined omega
        Omega = scipy.sparse.block_diag(Omegas).toarray()

        # Initialize z_hat
        if type(x0) is np.ndarray:
            z_hat = x0.copy()
        else:
            z_hat = z_bar.copy()

        for i in range(len(lc_dirs)):
            if lc_dirs[i] < 0:
                lcs[i] = self.invert_transform(lcs[i])

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


            A = Omega
            b = -Omega.dot(delta.flatten(order='F'))

            # Build up the loop closure part of the cost function
            for i in range(len(cycles)):
                this_edges = z_hat[:, cycles[i]]
                this_dirs = dirs[cycles[i]]
                this_lc = lcs[:, i, None]
                this_lc_omega = lc_omegas[i]

                # Create the mask to put the edges in their place when we are done
                mask = np.zeros((3*z_hat.shape[1], 3*this_edges.shape[1]))
                for j in range(len(cycles[i])):
                    mask[3*cycles[i][j]:3*cycles[i][j]+3,3*j:3*j+3] = np.eye(3)

                # Find Jacobian of cycle at this linearization point
                H_az = self.calc_jacobian_of_string_with_reversed_edges(this_edges, this_dirs)

                # concatenate edges
                z_long_way = self.compound_edges(this_edges, this_dirs)

                # Difference the edges (take care to handle the pi-wrap)
                residual = z_long_way - this_lc
                if residual[2] > np.pi:
                    residual[2] -= 2.0*np.pi
                if residual[2] <= -np.pi:
                    residual[2] += 2.0*np.pi

                A += mask.dot(H_az.T.dot(this_lc_omega).dot(H_az)).dot(mask.T)
                b -= mask.dot(H_az.T.dot(this_lc_omega).dot(residual).flatten())

            # Solve
            z_star = scipy.linalg.solve(A, b)

            z_star = z_star.reshape(z_hat.shape, order='F')

            z_hat += z_star

            diff = scipy.linalg.norm(z_star.flatten())
            print iter, ":", diff

            iter += 1

        # Update estimate
        return z_hat, diff

    def compound_arbitrary_edges(self, z, dirs, cycle):
        p = np.zeros((3,1))
        for i in cycle:
            if dirs[i] > 0:
                p += np.concatenate((self.R(p[2][0]).dot(z[0:2,i, None]), z[2,i,None, None]))
            else:
                p -= np.concatenate((self.R(p[2][0] - z[2,i]).dot(z[0:2,i, None]), z[2,i,None, None]))

            # wrap angle to +/- pi
            if p[2][0] > np.pi:
                p[2][0] -= 2.0*np.pi
            if p[2][0] <= -np.pi:
                p[2][0] += 2.0*np.pi

            # print p

        return p

    def compound_edges(self, z, dirs):
        return self.compound_arbitrary_edges(z, dirs, [i for i in range(len(dirs))])


    def calc_jacobian_of_string_with_reversed_edges(self, z, dirs):

        dtrans = z[0:2,:]
        thetas = z[2,:]

        num_edges = len(dirs)

        H = np.zeros((3, 3*len(dirs)))

        invert_indices = np.array([dirs < 0]).squeeze()

        # Calculate the angle of the translation partials
        # first, just a cumulative sum with a zero at the beginning (make sure to flip inverted edges)
        thetas[invert_indices] *= -1.0
        angles = np.concatenate((np.zeros(1), np.cumsum(thetas[0:-1])), axis=0)
        # inverted edges show up early, so add them
        angles[invert_indices] += thetas[invert_indices]

        # Create the rotation matrices and throw them into the jacobian (be sure to negate the inverted guys)
        # At the same time, calculate their rotations derivatives for later use
        Rd = np.zeros((2, num_edges))
        for i in range(num_edges):
            if dirs[i] > 0:
                H[0:2, 3*i:3*i + 2] = self.R(angles[i])
            else:
                H[0:2, 3*i:3*i + 2] = -self.R(angles[i])
            Rd[:,i] = (self.R(angles[i] + np.pi / 2.0).dot(dtrans[:, i, None])).squeeze()


        reversed_edges = np.zeros(num_edges)
        reversed_edges[invert_indices] -= 1.0
        mask2 = np.diag(dirs).dot(np.triu(np.matlib.repmat(dirs, num_edges, 1), 1) + np.diag(reversed_edges))

        g = (np.kron(mask2, np.eye(2)).dot(Rd.flatten(order='F'))).reshape(2,num_edges, order='F')

        for i in range(num_edges):
            H[0:2, 3*i + 2, None] = g[:,i, None]
            H[2, 3*i + 2] = dirs[i]
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

        optimizer.invert_edges(perfect_edges, dirs, [3, 4, 6])


        Omegas = [np.diag([1000., 1000., 10000.]) for i in range(perfect_edges.shape[1])]

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
        active_lc = [0]
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
    num_iters = 100
    zbar = edges.copy()
    iter = 0
    diff = 10000
    while iter < num_iters and diff > 0.00001:

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

        edges, diff = optimizer.optimize(zbar, dirs, Omegas, lc, lc_omega, lc_dir, cycles, 1, 0.000001, edges)
        iter += 1
        print "error = ", diff, "iters =", iter
    plt.show()

