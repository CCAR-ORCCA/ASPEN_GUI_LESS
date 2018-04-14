import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns





# Loading files
# P_X = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/Uncertainty/build/P_X.txt")
data = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/Uncertainty/build/data.txt")
simulated = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/Uncertainty/build/simulated.txt")


# 
# imgplot = plt.imshow(P_X)
# plt.colorbar()
# plt.show()

# The histograms are plotted
# (n,bins,patches) = plt.hist(data, 50, facecolor='green', alpha=0.5,label = "Ray-tracing")
# (n,bins,patches) = plt.hist(simulated, 50, facecolor='red', alpha=0.5,label = "Predicted")

sns.kdeplot(data,label = "Numerical",shade = True)
sns.kdeplot(simulated,label = "Analytical",shade = True)

plt.ylabel("Normalized occurences (-)")
plt.xlabel("Normalized range residuals (-)")
plt.legend(loc = "best")
# plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/conferences/GNSKi_2018/presentation/Figures/mc_results.pdf")
plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/papers/UQ_NAV_JGCD/R0/Figures/mc_results.pdf")

plt.show()
