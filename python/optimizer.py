from backend_optimizer._backend_optimizer_wrapper_cpp import BackendOptimizer

optimizer = BackendOptimizer()

nodes = [['0_000', 0, 1, 2], ['0_001', 3, 4, 5]]
edges = [1.0, 1.0]
fixed_node = 1

optimizer.new_graph(nodes, edges, fixed_node)

print(optimizer.get_optimized())