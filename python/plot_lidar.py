import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, ndimage
import os


def plot_lidar(path,savepath = None):


	focal_plane = np.loadtxt(path)

	plt.imshow(focal_plane,origin = "lower")
	plt.colorbar()


	plt.title(path[path.rfind('/') + 1:path.rfind('.')])
	if savepath is None:
		plt.show()
	else:
		plt.savefig(savepath)
	plt.clf()



# def plot_lidar(path_1,path_2,savepath = None):


# 	focal_plane_1 = np.loadtxt(path_1)
# 	focal_plane_2 = np.loadtxt(path_2)

# 	diff_focal_planes = focal_plane_1 - focal_plane_2


# 	std = np.std(np.ma.masked_invalid(diff_focal_planes))
# 	for i in range(diff_focal_planes.shape[0]):
# 		for j in range(diff_focal_planes.shape[1]):
# 			if abs(diff_focal_planes[i,j]) > 6 * std:
# 				diff_focal_planes[i,j] = float('NaN')

	

# 	plt.imshow(diff_focal_planes,origin = "lower")
# 	plt.colorbar()

# 	plt.show()
# 	plt.clf()


# plot_lidar("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/lidar/pc_true.txt")

# plot_lidar("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/lidar/pc_bezier.txt")

plot_lidar("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/lidar/pc_bezier_250000.000000.txt")
# plot_lidar("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/lidar/pc_true_250000.000000.txt")

