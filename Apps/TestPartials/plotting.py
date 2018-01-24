import numpy as np
import matplotlib.pyplot as plt

# Loading files
data = np.loadtxt("./build/data.txt")
simulated = np.loadtxt("./build/simulated.txt")

# The histograms are plotted
plt.hist(data, 50, normed=1, facecolor='green', alpha=0.5,label = "Ray-tracing")
plt.hist(simulated, 50, normed=1, facecolor='red', alpha=0.5,label = "Predicted")
plt.ylabel("Occurences")
plt.xlabel("Range distribution")
plt.legend(loc = "best")
# plt.show()
plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/status_report/Figures/mc_results.pdf")
