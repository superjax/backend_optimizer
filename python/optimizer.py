from backend_optimizer import backend_optimizer

list_to_sum = [1, 2, 3, 4, 5, 6, 7]

optimizer = backend_optimizer.BackendOptimizer()

print(optimizer.cumsum(list_to_sum))


debug = 1