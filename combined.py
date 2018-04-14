from REO import REO_opt
from GPO import GPO_opt

def combined_opt(edges, nodes, origin_node, num_iters, tol):
    # Single Iteration of REO
    x_hat_REO, iter_REO = REO_opt(edges, nodes, origin_node, num_iters, tol)

    # repackage nodes
    nodes_new = [ [nodes[i][0], x_hat_REO[0, i], x_hat_REO[1, i], x_hat_REO[2, i]] for i in xrange(len(nodes)) ]

    # Converge with GPO
    x_hat, iters = GPO_opt(edges, nodes_new, origin_node, num_iters, tol)
    # x_hat = x_hat_REO
    # iters = iter_REO
    return x_hat, iters