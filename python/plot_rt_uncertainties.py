import numpy as np
import matplotlib.pyplot as plt
from IOD_results_plots import draw_2d_covariance
path_to_folder = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/test_0"



X_errors = np.loadtxt(path_to_folder + "/X_error_arma.txt")
mrp_errors = np.loadtxt(path_to_folder + "/mrp_error_arma.txt")
R = np.loadtxt(path_to_folder + "/R_pcs_arma.txt")




plt.scatter(X_errors[:,1],X_errors[:,2],c = range(len(X_errors)))
for i in range(len(X_errors) - 1):
	draw_2d_covariance(0,R[i,:].reshape(6,6)[0:3,0:3],'b')
	plt.gca().annotate(str(i),(X_errors[i,1],X_errors[i,2]))

plt.xlabel("$e_y$")
plt.ylabel("$e_z$")

plt.title("X uncertainty, projected along $e_x$")
plt.axis("equal")
plt.show()

plt.clf()
plt.scatter(X_errors[:,0],X_errors[:,2],c = range(len(X_errors)))
for i in range(len(X_errors) - 1):
	draw_2d_covariance(1,R[i,:].reshape(6,6)[0:3,0:3],'b')
plt.title("X uncertainty, projected along $e_y$")

plt.xlabel("$e_x$")
plt.ylabel("$e_z$")

plt.axis("equal")
plt.show()

plt.clf()

plt.scatter(X_errors[:,0],X_errors[:,1],c = range(len(X_errors)))
for i in range(len(X_errors) - 1):
	draw_2d_covariance(2,R[i,:].reshape(6,6)[0:3,0:3],'b')
plt.title("X uncertainty, projected along $e_z$")

plt.xlabel("$e_x$")
plt.ylabel("$e_y$")
plt.axis("equal")
plt.show()

plt.clf()

plt.scatter(mrp_errors[:,0],mrp_errors[:,1],c = range(len(mrp_errors)))
for i in range(len(mrp_errors) - 1):
	draw_2d_covariance(2,R[i,:].reshape(6,6)[3:6,3:6],'b')

plt.axis("equal")
plt.show()


plt.clf()

plt.scatter(mrp_errors[:,0],mrp_errors[:,2],c = range(len(mrp_errors)))
for i in range(len(mrp_errors) - 1):
	draw_2d_covariance(1,R[i,:].reshape(6,6)[3:6,3:6],'b')


plt.axis("equal")
plt.show()


plt.clf()

plt.scatter(mrp_errors[:,1],mrp_errors[:,2],c = range(len(mrp_errors)))
for i in range(len(mrp_errors) - 1):
	draw_2d_covariance(0,R[i,:].reshape(6,6)[3:6,3:6],'b')

plt.axis("equal")
plt.show()