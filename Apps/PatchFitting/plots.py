import numpy as np
import matplotlib.pyplot as plt

results = np.loadtxt("output/results.txt")

plt.scatter(range(results.shape[1]),results[1,:] / results[0,:],marker = ".")
plt.show()