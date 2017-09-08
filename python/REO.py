import numpy as np
import scipy.sparse
import scipy.linalg

class REO():
    def __init__(self):
        debug = 1

    def optimize(self, z_bar, dirs, Omegas, lc, lc_omega, lc_dir, iters):

        # create giant combined omega
        Omega = scipy.sparse.block_diag(Omegas)

        # Initialize z_hat
        z_hat = z_bar


        for i in range(iters):
            # Find Jacobian of cycle at this linearization point
            H = self.calc_jacobian_of_string_with_reversed_edges(z_hat, dirs)

            # Calculate left side
            A = Omega + H.T.dot(lc_omega).dot(H)

            # concatenate edges
            z_long_way = self.compound_edges(z_hat, dirs)

            # Calculate right side
            b = H.T.dot(lc_omega).dot(z_long_way - lc)

            # Solve
            z_star = scipy.linalg.solve(A, b)

            # Update estimate
            z_hat -= z_star.reshape(z_hat.shape, order='F')
        return z_hat



    def compound_edges(self, z, dirs):
        p = np.zeros((3,1))
        for i in range(z.shape[1]):
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



    def calc_jacobian_of_string_with_reversed_edges(self, z, dirs):

        dtrans = z[0:2,:]
        theta = z[2,:]

        H = np.zeros((3, 3*len(dirs)))

        total_angle_so_far = 0
        for i in range(len(dirs)):

            if dirs[i] > 0:
                H[0:2, 3*i:3*i + 2] = self.R(total_angle_so_far)
            else:
                H[0:2, 3*i:3*i + 2] = -1.0*self.R(total_angle_so_far + theta[i])

            total_angle_so_far += theta[i]

        for i in range(len(dirs)):

            dt_dtheta = np.zeros((2,1))
            for j in range(i):
                dt_dtheta += H[0:2, 3*j:3*j + 2].dot(dtrans[:,j, None])
            if dirs[i] < 0:
                dt_dtheta += H[0:2, 3*i:3*i + 2].dot(dtrans[:,i, None])
                dt_dtheta *= -1.0
            H[0:2, 3*i + 2, None] = dt_dtheta
            H[2, 3*i + 2] = dirs[i]
        return H


    def R(self,theta):
        ct = np.cos(theta)
        st = np.sin(theta)
        return np.array([[ct, -st],
                         [st,  ct]])

def concatenate_transform(T1, T2):
    x = T1[0] + T2[0] * np.cos(T1[2]) - T2[1] * np.sin(T1[2])
    y = T1[1] + T2[0] * np.sin(T1[2]) + T2[1] * np.cos(T1[2])
    psi = T1[2] + T2[2]
    return [x, y, psi]

def invert_transform(T):
    dx = -(   T[0]*np.cos(T[2]) + T[1]*np.sin(T[2]))
    dy = -( - T[0]*np.sin(T[2]) + T[1]*np.cos(T[2]))
    psi = -T[2]
    return [dx, dy, psi]

def invert_edges(x, dirs, indexes):
    for index in indexes:
        x[:,index] = invert_transform(x[:,index])
        dirs[index] *= -1.0


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    optimizer = REO()

    perfect_edges = np.array([[1., 1., 1., 1., 2**0.5, 2**0.5/2.0, 2**0.5/2.0, 2**0.5],
                              [0., 0., 0., 0., 0,      0,          0,          0],
                              [np.pi/2.0, np.pi/2.0, np.pi/2.0, 3.0*np.pi/4.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, 0]])

    dirs = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    invert_edges(perfect_edges, dirs, [3, 4, 6])


    Omegas = [np.diag([10., 10., 10.]) for i in range(perfect_edges.shape[1])]

    edge_noise = np.array([[np.random.normal(0, 1./Omegas[i][0][0]) for i in range(perfect_edges.shape[1])],
                           [np.random.normal(0, 1./Omegas[i][1][1]) for i in range(perfect_edges.shape[1])],
                           [np.random.normal(0, 1./Omegas[i][2][2]) for i in range(perfect_edges.shape[1])]])

    noisy_edges = perfect_edges + edge_noise

    lc = np.array([[1.0],
                   [0.0],
                   [0.0]])
    lc_omega = np.diag([1000, 1000, 0])
    lc_dir = 1

    edges = perfect_edges
    edges = noisy_edges

    plt.ion()
    for iter in range(100):

        global_pose = np.zeros((3, noisy_edges.shape[1]+1))
        for i in range(edges.shape[1]):
            if dirs[i] > 0:
                global_pose[:,i+1] = np.array(concatenate_transform(global_pose[:,i], edges[:,i]))
            else:
                global_pose[:, i + 1] = np.array(concatenate_transform(global_pose[:, i], invert_transform(edges[:, i])))

        global_lc_location = np.zeros((3,2))
        global_lc_location[:,0] = global_pose[:,0]
        global_lc_location[:,1] = np.array(concatenate_transform(global_pose[:,0], lc[:,0]))

        plt.figure(0)
        plt.clf()
        plt.title('iteration ' + str(iter))
        plt.plot(global_pose[1,:], global_pose[0,:], 'b')
        plt.plot(global_lc_location[1, :], global_lc_location[0, :], 'r')
        plt.show()

        debug = 1

        edges = optimizer.optimize(edges, dirs, Omegas, lc, lc_omega, lc_dir, 1)

