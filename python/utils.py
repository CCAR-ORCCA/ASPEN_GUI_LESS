import numpy as np
import matplotlib.pyplot as plt

def plot_focal_plane():

	ranges = np.loadtxt("../build/ranges.txt")
	cm = plt.cm.get_cmap('RdYlBu')

	y_res = np.amax(ranges[:,0]) + 1;
	z_res = np.amax(ranges[:,1]) + 1;

	min_range = min([ranges[i,-1] for i in range(len(ranges[:,-1])) if ranges[i,-1] < 1e20])
	max_range = max([ranges[i,-1] for i in range(len(ranges[:,-1])) if ranges[i,-1] < 1e20])

	valid_mes = np.vstack([ranges[i,:]for i in range(len(ranges[:,-1])) if ranges[i,-1] < 1e20])
	if len(valid_mes) != len(ranges):
		invalid_mes = np.vstack([ranges[i,:]for i in range(len(ranges[:,-1])) if ranges[i,-1] > 1e20])

	sc = plt.scatter(valid_mes[:,0],valid_mes[:,1],c = valid_mes[:,2], 
		s = 60, vmin = min_range, vmax = max_range, marker = "s",cmap = cm)
	
	plt.colorbar(sc)
	plt.xlim([0,y_res - 1])
	plt.ylim([0,z_res - 1])

	plt.grid()
	plt.show()

def plot_long_lat():

	long_lat = np.loadtxt("../output/long_lat.txt")
	long_lat_rel = np.loadtxt("../output/long_lat_rel.txt")

	plt.scatter(180 / np.pi * long_lat[:,0],180 / np.pi * long_lat[:,1],c= 'b',label = 'Inertial frame')
	plt.scatter(180 / np.pi * long_lat_rel[:,0],180 / np.pi * long_lat_rel[:,1],c= 'r',label = 'Body-fixed frame')
	plt.legend(bbox_to_anchor=(0.5, 1.1),ncol = 2,loc = 'upper center')
	plt.grid()
	plt.xlim([-180,180])
	plt.ylim([-90,90])
	plt.xlabel("Longitude (deg)")
	plt.ylabel("Latitude (deg)")
	plt.savefig("long_lat.pdf")
	plt.show()

def plot_diff():
	volume_dif = np.loadtxt("../output/volume_dif.txt")
	surface_dif = np.loadtxt("../output/surface_dif.txt")

	plt.plot(range(len(volume_dif)),100 * np.abs(volume_dif),label = 'Volume')
	plt.plot(range(len(surface_dif)),100 * np.abs(surface_dif),label = 'Area')
	plt.legend(bbox_to_anchor=(0.5, 1.1),ncol = 2,loc = 'upper center')
	plt.ylabel("Relative difference (%)")
	plt.xlabel("Measurement index")


	plt.savefig("dif.pdf")



def plot_volume_dif():
	volume_dif = np.loadtxt("../output/volume_dif.txt")

	plt.plot(range(len(volume_dif)),100 * np.abs(volume_dif))
	plt.xlabel("Measurement index")
	plt.ylabel("Relative volume difference (%)")
	plt.savefig("volume_dif.pdf")


	plt.show()

def plot_surface_dif():
	surface_dif = np.loadtxt("../output/surface_dif.txt")

	plt.plot(range(len(surface_dif)),100 * np.abs(surface_dif))
	plt.xlabel("Measurement index")
	plt.ylabel("Relative surface difference (%)")
	plt.savefig("surface_dif.pdf")


	plt.show()






