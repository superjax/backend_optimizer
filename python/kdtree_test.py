from backend_optimizer import kdtree
import random
import numpy as np
import time
import scipy.spatial

new_tree = kdtree.KDTree()


# Create list of points to test with
i = 0
points = []
for i in range(100000):
    new_line = ['0_'+str(i)]
    new_line.extend([100*random.random() for i in range(3)])
    points.append(new_line)

points[5] = ['0_5', 0.5, 0.5, 0.5]

start = time.time()
new_tree.add_points(points)
end = time.time()

print "took", end - start, "seconds for nanoflann to sort 1"

start = time.time()
coordinates = [j[1:] for j in points]
scipy_tree = scipy.spatial.KDTree(coordinates)
end = time.time()

print "took", end-start, "seconds for scipy to sort 1"

start = time.time()
# for i in range(1000):
query_point = ['0_5', 0.5, 0.5, 0.5]
closest_index = new_tree.find_closest_point(query_point, 25)
end = time.time(    )
print "took nanoflann", end - start, "seconds to find closest points"


start = time.time()
loop_closures = scipy_tree.query_ball_point([0.5, 0.5, 0.5], 25)
closest_scipy_point = 'none'
for index in loop_closures:
    to_node_id = points[index][0]

    if to_node_id == '0_5':
        continue  # Don't calculate stupid loop closures

    closest_scipy_point = to_node_id
end = time.time()
print "took scipy", end - start, "seconds to find closest points"


del(points[5])

min = 10
min_index = 0
for i in points:
    dist =  np.linalg.norm(np.array(i[1:]) - np.array(query_point[1:]))
    if dist < min:
        min = dist
        min_index = i[0]
    # print i[0], dist

print "actual closest =", min_index, min

print "nanoflann closest", closest_index, np.linalg.norm(np.array(closest_index[1:]) - np.array(query_point[1:]))
print "scipy closest", closest_scipy_point



