import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, ndimage



def plot_lidar(path,savepath = None):


	focal_plane = np.loadtxt(path)

	plt.imshow(focal_plane,origin = "lower")
	plt.colorbar()

	if savepath is None:
		plt.show()
	else:
		plt.savefig(savepath)
	plt.clf()


def plot_fft(path,savepath = None):


	focal_plane = np.loadtxt(path)

	plt.imshow(np.log(np.abs(np.fft.fftshift(focal_plane))**2))
		
	plt.colorbar()

	if savepath is None:
		plt.show()
	else:
		plt.savefig(savepath)
	plt.clf()



plot_fft("../Apps/Navigation/build/focal_plane_true_0.txt")


plot_lidar("../Apps/Navigation/build/focal_plane_true_862068.txt")