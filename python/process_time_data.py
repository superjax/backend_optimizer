import pickle
import matplotlib.pyplot as plt
import numpy as np

data = np.array(pickle.load(open("time_array.pkl", "rb")))

id = data[:,0]
num_graphs = data[:,1]
num_nodes_in_graph = data[:,2]
num_new_nodes = data[:,3]
num_new_edges = data[:,4]
time_ms = data[:,5]

debug = 1

print "average", np.average(time_ms)
print "max", np.max(time_ms)
print "min", np.min(time_ms)
print "var", np.var(time_ms)
print "sum", sum(time_ms)/60.0

plt.figure(1)
plt.title("time vs num_nodes_in_graph")
plt.plot(num_nodes_in_graph, time_ms, '.')
plt.xlabel('num nodes')
plt.ylabel('ms')

plt.figure(2)
plt.title("optimization")
plt.plot(time_ms)
plt.ylabel('ms')

plt.figure(3)
plt.title("num_nodes_in_graph")
plt.plot(num_nodes_in_graph)
plt.ylabel('num_nodes_in_graph')

plt.figure(4)
plt.title("time vs new nodes")
plt.plot(num_new_nodes, time_ms, '.')
plt.xlabel('num new nodes')
plt.ylabel('ms')

plt.figure(5)
plt.title("time vs new edges")
plt.plot(num_new_edges, time_ms, '.')
plt.xlabel('num new edges')
plt.ylabel('ms')



plt.show()


