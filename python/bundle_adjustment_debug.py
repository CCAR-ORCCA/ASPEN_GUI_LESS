import numpy as np
import matplotlib.pyplot as plt

N = 6

X_tilde_before_ba = []
X_tilde_after_ba = []
X_tilde_true = []


sigma_tilde_before_ba = []
sigma_tilde_after_ba = []
sigma_tilde_true = []

path = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/IOD/output/transforms/"

for i in range(1,N):

	X_tilde_before_ba += [np.loadtxt(path + "X_tilde_before_ba_"+ str(i) +".txt")]
	X_tilde_after_ba += [np.loadtxt(path + "X_tilde_after_ba_"+ str(i) +".txt")]
	X_tilde_true +=[np.loadtxt(path + "X_tilde_true_"+ str(i) +".txt")]

	sigma_tilde_before_ba += [np.loadtxt(path + "sigma_tilde_before_ba_"+ str(i) +".txt")]
	sigma_tilde_after_ba += [np.loadtxt(path + "sigma_tilde_after_ba_"+ str(i) +".txt")]
	sigma_tilde_true += [np.loadtxt(path + "sigma_tilde_true_"+ str(i) +".txt")]

X_tilde_before_ba = np.vstack(X_tilde_before_ba)
X_tilde_after_ba = np.vstack(X_tilde_after_ba)
X_tilde_true = np.vstack(X_tilde_true)

sigma_tilde_before_ba = np.vstack(sigma_tilde_before_ba)
sigma_tilde_after_ba = np.vstack(sigma_tilde_after_ba)
sigma_tilde_true = np.vstack(sigma_tilde_true)


plt.scatter(range(1,N),np.linalg.norm(X_tilde_before_ba - X_tilde_true,axis = 1),label = "before BA")
plt.scatter(range(1,N),np.linalg.norm(X_tilde_after_ba - X_tilde_true,axis = 1),label = "after BA")
plt.legend(loc = "best")
plt.title("Translation error")
plt.show()
plt.clf()

plt.scatter(range(1,N),np.linalg.norm(sigma_tilde_before_ba - sigma_tilde_true,axis = 1),label = "before BA")
plt.scatter(range(1,N),np.linalg.norm(sigma_tilde_after_ba - sigma_tilde_true,axis = 1),label = "after BA")
plt.legend(loc = "best")
plt.title("MRP error")
plt.show()