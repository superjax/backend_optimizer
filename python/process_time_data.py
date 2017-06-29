import pickle
import matplotlib.pyplot as plt
import numpy as np

data = pickle.load(open("time_array.pkl", "rb"))

debug = 1

print "average", np.average(data)
print "max", np.max(data)
print "min", np.min(data)
print "var", np.var(data)
print "sum", sum(data)/60.0

plt.figure(1)
plt.plot(data)
plt.show()
