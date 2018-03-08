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

# plot_lidar("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/Navigation/output/lidar/pc_true.txt")

plot_lidar("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/Navigation/output/lidar/pc_bezier.txt")
