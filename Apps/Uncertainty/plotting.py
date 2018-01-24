import numpy as np
import matplotlib.pyplot as plt






# Loading files
P_CC = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/Uncertainty/build/P_CC.txt")
data = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/Uncertainty/build/data.txt")
simulated = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/Uncertainty/build/simulated.txt")


# 
imgplot = plt.imshow(P_CC)
plt.colorbar()
plt.show()

# The histograms are plotted
(n,bins,patches) = plt.hist(data, 50, facecolor='green', alpha=0.5,label = "Ray-tracing")
(n,bins,patches) = plt.hist(simulated, 50, facecolor='red', alpha=0.5,label = "Predicted")
plt.ylabel("Occurences")
plt.xlabel("Range distribution")
plt.legend(loc = "best")
plt.show()
# plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/status_report/Figures/mc_results.pdf")
