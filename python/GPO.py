import numpy as np
import backend_optimizer

class GPO:
    def __init__(self):
        self.GPO = backend_optimizer.Optimizer()

    def opt(self, edges, nodes, origin_node, num_iters, tol):
        out_dict = self.GPO.batch_optimize(nodes, edges, origin_node, num_iters, tol)
        opt_pose = []
        for node in out_dict['nodes']:
            opt_pose.append(node[1:])
        opt_pose = np.array(opt_pose)
        return opt_pose.T.copy(), out_dict['iter']
