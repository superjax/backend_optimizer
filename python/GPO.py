import numpy as np
import backend_optimizer

def GPO_opt(edges, nodes, origin_node, num_iters, tol):
    GPO = backend_optimizer.Optimizer()
    out_dict = GPO.batch_optimize(nodes, edges, origin_node, num_iters, tol)
    opt_pose = []
    for node in out_dict['nodes']:
        opt_pose.append(node[1:])
    opt_pose = np.array(opt_pose)
    return opt_pose.T.copy(), out_dict['iter']
